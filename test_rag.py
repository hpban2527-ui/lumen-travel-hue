import json
import math
import pytest
from guide_loader import load_guide
from rag import (ABSTAIN, RAGError, retrieve, embed_texts,
                 validate_generation, clean_base_url, answer_question, Match)


class FakeReply:
    def __init__(self, data, code=200): self.data=data;self.status_code=code
    def json(self): return self.data


class Transport:
    def __init__(self, answer=None):
        self.calls=[]
        self.answer=answer or json.dumps({'supported':True,'answer':'The river is the Perfume River. [G05]','citations':['G05']})
    def post(self, url, **kwargs):
        self.calls.append((url,kwargs))
        if url.endswith('/api/embed'):
            count=len(kwargs['json']['input'])
            return FakeReply({'embeddings':[[1.,0.,0.] for _ in range(count)]})
        return FakeReply({'message':{'content':self.answer}})


def test_cosine_order_and_top_k():
    entries=load_guide()[:3]
    guide_vectors=((1.,0.), (0.,1.), (.6,.8))
    hits=retrieve((1.,0.),entries,guide_vectors,k=2)
    assert [x.entry['id'] for x in hits]==['G01','G03']
    assert hits[0].score==pytest.approx(1)
    assert hits[1].score==pytest.approx(.6)


def test_embedding_batch_and_model():
    t=Transport()
    vectors=embed_texts('http://localhost:11434',['a','b'],transport=t)
    assert len(vectors)==2
    assert t.calls[0][1]['json']['model']=='embeddinggemma'


def test_malformed_and_dimension_errors():
    with pytest.raises(RAGError,match='dimensions differ'):
        retrieve([1,0],load_guide()[:1],[[1,0,0]],k=1)
    with pytest.raises(RAGError,match='zero vector'):
        retrieve([0,0],load_guide()[:1],[[1,0]],k=1)


def test_guardrail_accepts_genuine_listed_citation():
    matches=[Match(load_guide()[4],0.88)]
    out=validate_generation(json.dumps({'supported':True,'answer':'Perfume River crosses Huế. [G05]','citations':['G05']}),matches)
    assert out.supported and '[G05]' in out.text


@pytest.mark.parametrize('response',[
  {'supported':True,'answer':'The ticket is $20 [G99]','citations':['G99']},
  {'supported':True,'answer':'The ticket is $20','citations':['G05']},
  {'supported':False,'answer':"I don't know",'citations':[]},
  {'supported':True,'answer':'No citations','citations':[]},
])
def test_guardrail_abstains_on_unsupported_answers(response):
    out=validate_generation(json.dumps(response),[Match(load_guide()[4],0.5)])
    assert not out.supported and out.text==ABSTAIN


def test_think_data_not_shown():
    out=validate_generation('<think>hidden</think>\n'+json.dumps({'supported':False,'answer':'I do not know','citations':[]}),[Match(load_guide()[0],0.5)])
    assert not out.supported and 'hidden' not in out.text


def test_full_pipeline_api_calls_use_required_models():
    t=Transport()
    guide=load_guide()
    vector=[(1.,0.,0.) if e['id']=='G05' else (0.,1.,0.) for e in guide]
    out=answer_question('http://localhost:11434','Which river?',guide,vector,transport=t)
    assert out.supported and out.matches
    assert t.calls[0][1]['json']['model']=='embeddinggemma'
    assert t.calls[1][1]['json']['model']=='qwen3:4b'
    assert t.calls[1][1]['json']['format']=='json'


def test_remote_http_rejected():
    with pytest.raises(RAGError,match='HTTPS'):
        clean_base_url('http://example.com')


def test_blank_question():
    with pytest.raises(RAGError,match='Type a question'):
        answer_question('http://localhost:11434',' ',load_guide(),[(1.,0.)]*16)

@pytest.mark.parametrize('question',[
    'What is the ticket price today at the Imperial City?',
    'Which tombs are open at 9pm tomorrow?',
    'What is the weather in Huế right now?',
])
def test_volatile_guide_gaps_abstain_without_generation(question):
    t=Transport()
    guide=load_guide()
    vectors=[(1.,0.,0.) if e['id']=='G05' else (0.,1.,0.) for e in guide]
    result=answer_question('http://localhost:11434',question,guide,vectors,transport=t)
    assert not result.supported and result.text==ABSTAIN
    assert len(t.calls)==1  # embed query, skip generation
