# Lumen Travel — Huế City Guide

**Computational Thinking · Practice 03**  
**Student:** Huỳnh Phương Bình An · **Student ID:** 25127009 · **Class:** 25C09  
**City:** Huế, Việt Nam

Lumen Travel is a small travel assistant built around a curated Huế guide. Sign in with an email and password, ask a question about the city, and see both the answer and the guide entries behind it. When the guide has no relevant answer, the app says so rather than supplying an invented opening time or price.

## Features

- Firebase Authentication: create an account, sign in, and sign out.
- Private chat interface available only after sign-in, with history held in `st.session_state` and cleared on sign-out.
- Sixteen short guide entries in `knowledge_base.json`, each with an ID, title, text, source, category, and tags.
- Retrieval with `embeddinggemma` and cosine similarity (top three entries).
- Context-restricted generation with `qwen3:4b`, cited guide IDs (`[G01]`, etc.), and expandable retrieval details with scores.
- Cached guide vectors and readable errors for invalid input, authentication failures, and server connection issues.

## Project files

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface and session workflow |
| `firebase_client.py` | Firebase email/password authentication |
| `guide_loader.py` | Guide validation and loading |
| `rag.py` | Embedding, retrieval, prompt context, and answer validation |
| `knowledge_base.json` | Huế travel guide |
| `evaluate.py` | Six-question evaluation runner |
| `styles.css` | Application styling |
| `tests/` | Unit tests for authentication, guide data, retrieval, and evaluation |

## Run locally

1. Install **Python 3.10+** and [Ollama](https://ollama.com/). Enable Email/Password sign-in in your Firebase Authentication project.
2. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Pull the two models on your Ollama server:

   ```bash
   ollama pull embeddinggemma
   ollama pull qwen3:4b
   ```

4. Copy `.env.example` to `.env`, then set `FIREBASE_API_KEY` to the Firebase Web API key and `OLLAMA_BASE_URL` to the Ollama API base URL exposed through the Lab 4 Pinggy tunnel. Use `http://localhost:11434` when Ollama runs locally. Keep `.env` out of version control.
5. Start the app:

   ```bash
   streamlit run app.py
   ```

## Computational Thinking

**Abstraction.** A question and a signed-in session are the inputs; an answer with guide citations, similarity results, and chat history is the output. Guide records keep their identifier, title, description, original source, category, and tags: the first four support traceable answers, while the final two organize places and prepare for RAG-Rec.

**Decomposition.** `Firebase sign-in → question embedding → cosine top-k retrieval → cited context → qwen3:4b answer → evidence display`. Authentication gates the chat; the retrieval and generation stages remain separate to make errors easier to locate.

**Pattern recognition.** The app combines Streamlit and Firebase from Lab 3 with Ollama embedding and generation from Lab 4. The `score → top-k` pattern selects the guide passages most relevant to each question. A later recommendation system can reuse scoring and ranking with category or preference filters.

**Algorithm design.** For a non-empty question: check the signed-in session; embed the question; calculate cosine similarity against cached guide vectors; select three entries; give only those entries to the chat model; accept answers with allowed citations, otherwise return “I don't know”; display the answer and retrieved entries. Three results keep the context compact while allowing a question to match more than one guide topic.

## Six-question evaluation

The six questions below cover three in-guide facts, one paraphrase, and two facts outside the guide. Cosine scores are rounded to three decimal places. The HTTP 500 response is recorded as an integration failure, not a correct answer.

To repeat the evaluation against a configured Ollama server:

```bash
python evaluate.py --out evaluation_results.json
```

| Type | Question | Expected entry | Retrieved IDs (cosine score) | Correct? | Failure stage |
|---|---|---|---|---|---|
| Answerable | What is the Imperial City in Huế? | G01 | — (HTTP 500) | No | Integration / Ollama API |
| Answerable | Tell me about Dong Ba Market. | G03 | G03 (0.727), G14 (0.373), G09 (0.345) | Yes | — |
| Answerable | Which river flows through Huế? | G05 | G05 (0.664), G04 (0.601), G02 (0.523) | Yes | — |
| Paraphrase | Where can I learn about Nguyen royal court history? | G01 / G11 / G16 | G11 (0.585), G01 (0.517), G12 (0.512) | Yes | — |
| Outside guide | What is the ticket price today at the Imperial City? | I don't know | G01 (0.357), G11 (0.259), G06 (0.244) | Yes | — |
| Outside guide | Which tombs will be open at 9 p.m. tomorrow? | I don't know | G07 (0.344), G06 (0.316), G08 (0.308) | Yes | — |

**Result:** 5 of 6 cases met the stated test criteria. The HTTP 500 request was an integration error; its underlying server cause was not established from this run. The two outside-guide questions correctly produced a refusal rather than a guessed price or schedule.

## Scope and limitations

The app uses a fixed set of guide notes, not a live travel feed. It does not look up current prices, special closures, or daily schedules. Cosine similarity ranks passages by semantic closeness, not factual certainty; the citation check narrows the scope of answers but does not independently establish whether every claim is true. Chat history is limited to the current session and is not stored in a database.

##AI tools used
AI tools were used only for editing and animating the demonstration video, and for advice on how to screen-record the app and capture short footage quickly. The application code was written by me.
Test results are reported only as measured: the offline tests ran with mocked services, and live results are entered in section 3.2 only after a real run.


## Demonstration video

Video (YouTube): https://youtu.be/jLjsoR_4dUo

Source repository (GitHub): https://github.com/hpban2527-ui/lumen-travel-hue

