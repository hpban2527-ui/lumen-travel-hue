from copy import deepcopy
import json
import pytest
from guide_loader import GuideError, load_guide, guide_digest


def test_real_guide_has_16_well_formed_entries():
    guide=load_guide()
    assert len(guide)==16
    assert guide[0]['id']=='G01'
    assert guide_digest()


def test_duplicate_id_rejected(tmp_path):
    entries=deepcopy(load_guide())
    entries[1]['id']=entries[0]['id']
    file=tmp_path/'guide.json'
    file.write_text(json.dumps(entries))
    with pytest.raises(GuideError, match='Duplicate'):
        load_guide(file)


def test_missing_source_rejected(tmp_path):
    entries=deepcopy(load_guide())
    entries[0]['source']='not-a-url'
    file=tmp_path/'guide.json'
    file.write_text(json.dumps(entries))
    with pytest.raises(GuideError, match='source URL'):
        load_guide(file)
