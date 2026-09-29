# Structured Output এবং Function Calling

**Phase:** [Prompt Engineering and In-Context Learning](../README.md) · **Topic folder:** `05-Structured-Output-and-Function-Calling`

## কেন এটি গুরুত্বপূর্ণ

এই phase-এর এখন পর্যন্ত প্রতিটি কৌশল — [few-shot example](../01-Prompting-Basics-Zero-Few-Shot/README.md), [chain-of-thought](../02-Chain-of-Thought-and-Reasoning-Prompts/README.md), [ReAct](../03-Tree-of-Thought-and-ReAct/README.md), এমনকি [স্বয়ংক্রিয়ভাবে আরও ভালো prompt খোঁজা](../04-Automatic-Prompt-Optimization/README.md) — এখনও শুধু model-কে ভদ্রভাবে একটি নির্দিষ্ট ধরনের output তৈরি করতে *অনুরোধ* করে। Output যখন একজন মানুষ পড়বে এমন free-form text, তখন এটা ঠিক আছে, কিন্তু যে মুহূর্তে downstream code-কে response-টি `json.loads()` করতে হয়, বা নির্দিষ্ট argument দিয়ে একটি নির্দিষ্ট function call করতে হয়, তখন "model-কে JSON output দিতে নির্দেশ দেওয়া হয়েছিল" কোনো নিশ্চয়তা নয় — এটি একটি আশা। এই lesson সেই দুটি কৌশল কভার করে যা সেই আশাকে নিশ্চয়তার অনেক কাছাকাছি কিছুতে পরিণত করে: **constrained decoding**, যা syntactic validity লঙ্ঘন করাকে কাঠামোগতভাবে অসম্ভব করে তোলে, এবং **function calling**, সেই request → parse → execute → respond loop যা একটি model-এর structured output-কে বাস্তব জগতে সত্যিই কিছু *করতে* দেয় — ঠিক সেই mechanism যার উপর [Lesson 3-এর ReAct loop](../03-Tree-of-Thought-and-ReAct/README.md#3-react-yao-et-al-2022-reason-act-observe-repeat) প্রতিবার একটি "Act" step নেওয়ার সময় নির্ভর করে।

## এই lesson যা কভার করে

- কেন শুধু prompting বৈধ structured output-এর *নিশ্চয়তা* দিতে পারে না, model যত ভালোই হোক না কেন
- Constrained decoding: প্রতিটি sampling step-এর আগে অবৈধ token-কে `-inf`-এ mask করা
- একটি প্রকৃত JSON grammar, character-level state machine হিসেবে implement করা
- Experiment দিয়ে প্রমাণ: একটি সত্যিকারের untrained, randomly-initialized model, তবুও 100% syntactically valid হতে বাধ্য
- Function calling: `{"name": ..., "arguments": {...}}` request format
- পূর্ণ request → parse → execute → respond loop, প্রকৃত Python function সত্যিই চলছে
- ReAct-এর তুলনায় এটি কোথায় বসে: function calling হলো mechanism, আর ReAct হলো কখন এটি invoke করতে হবে তা ঠিক করার একটি strategy

## 1. কেন "শুধু ভদ্রভাবে অনুরোধ করা" যথেষ্ট নয়

একটি ভালোভাবে trained, ভালোভাবে prompt করা model বেশিরভাগ সময়েই বৈধ JSON output দেয় — কিন্তু "বেশিরভাগ সময়" এমন code-এর জন্য যথেষ্ট নয় যা response-এ `json.loads()` call করে এবং ব্যর্থ ক্ষেত্রগুলোতে crash করে (বা আরও খারাপ, নিঃশব্দে ভুল আচরণ করে)। একটি বিপথগামী trailing comma, একটি string-এর ভেতরে একটি unescaped quote, অথবা object-এর মাঝখানে কেটে যাওয়া একটি truncated response একটি naive parser-কে ভেঙে দেয়। দুটি কাঠামোগতভাবে ভিন্ন সমাধান আছে: *generation process-টিকেই* constrain করো যাতে অবৈধ token কখনো তৈরি না হয় (`example.py`-এর Part A), অথবা মেনে নাও যে model-এর raw output শুধু একটি *request* এবং এর অপর পাশে একটি প্রকৃত, রক্ষণাত্মক execution layer রাখো (Part B)। Production system সাধারণত দুটিই একসাথে ব্যবহার করে।

## 2. Constrained decoding: token level-এ validity-র নিশ্চয়তা

প্রতিটি generation step-এ একটি language model তার পুরো vocabulary-র উপর একটি probability distribution (logits) তৈরি করে। Constrained decoding **sampling-এর আগে** হস্তক্ষেপ করে: এখন পর্যন্ত যা তৈরি হয়েছে তা দেওয়া থাকলে, পরবর্তীতে আসা grammatically বৈধ character/token-এর set গণনা করো, এবং sampling-এর আগে বাকি প্রতিটি logit-কে `-infinity`-তে set করো (যাতে এর post-softmax probability ঠিক শূন্য হয়):

```
logits = model(generated_so_far)          # raw scores over the whole vocabulary
valid  = grammar.next_valid_tokens(generated_so_far)   # e.g. {'"', 'a', 'd', ...}
for token in vocabulary:
    if token not in valid:
        logits[token] = -inf
next_token = sample(softmax(logits))
```

যেহেতু প্রতিটি অবৈধ token-এ শূন্য probability-যুক্ত একটি distribution থেকে sampling *কখনোই* সেগুলোর একটিকে বেছে নিতে পারে না, এটি একটি **কঠোর নিশ্চয়তা**, কোনো statistical প্রবণতা নয় — এবং গুরুত্বপূর্ণভাবে, সেই নিশ্চয়তা **underlying model যত ভালো বা খারাপই হোক** বজায় থাকে। `example.py` Part A এটি সম্ভাব্য সবচেয়ে চরম উপায়ে প্রমাণ করে: এটি এমন একটি Transformer-এ constrained decoding প্রয়োগ করে যা কখনো train-ই করা হয়নি (বিশুদ্ধ random weights) এবং তবুও প্রতিবার নিখুঁতভাবে বৈধ output পায়।

## 3. Grammar, একটি state machine হিসেবে

`example.py` একটি JSON shape-এর জন্য একটি ছোট কিন্তু প্রকৃত grammar সংজ্ঞায়িত করে:

```
{"action": "add" | "sub" | "mul", "value": <1-to-3-digit integer, no leading zero>}
```

`next_valid_chars(prefix)` এটিকে একটি character-level state machine হিসেবে implement করে: শুধু *এখন পর্যন্ত* তৈরি string দেওয়া থাকলে, এটি পরবর্তীতে emit করার জন্য বৈধ character-গুলোর হুবহু set ফেরত দেয় (একটি খালি set মানে structure ইতিমধ্যে সম্পূর্ণ — generation অবশ্যই থামাতে হবে)। যেমন, quote-এর ভেতরে `rest` একবার হুবহু `"add"`-এর সাথে মিলে গেলে, একমাত্র বৈধ পরবর্তী character হলো closing quote; `value`-এর জন্য তিনটি digit emit হয়ে গেলে, একমাত্র বৈধ পরবর্তী character হলো `}`। Script-টি একটি **দ্বিতীয়, স্বাধীনভাবে লেখা** full-string validity checker-ও implement করে এবং grammar-এর মধ্য দিয়ে হাজার হাজার random walk এর বিরুদ্ধে cross-check করে — এটি সেই একই "mechanism-টি যা দাবি করে তা সত্যিই করে কিনা যাচাই করো" শৃঙ্খলা যা এই পুরো কোর্স জুড়ে ব্যবহৃত, শুধু দাবি করা নয়।

## 4. Experiment: validity আসে masking থেকে, learning থেকে নয়

Grammar যুক্ত করার পর, `example.py` Part A একটি ক্ষুদ্র, সত্যিকারের **untrained** causal Transformer (random initialization, Part A-র কোথাও কোনো training loop নেই) থেকে দুইভাবে generate করে: একবার প্রতিটি step-এ grammar mask প্রয়োগ করে, একবার কোনো mask ছাড়াই (পুরো character vocabulary-র উপর free sampling)। ফলাফল: **constrained generation-এর 100% বৈধ**, অথচ হুবহু একই random model থেকে unconstrained generation প্রায় কখনোই বৈধ নয়। যেহেতু দুই ক্ষেত্রেই model-এর weights বিশুদ্ধ noise, validity-র সেই ব্যবধানের কিছুই model যা "জানে" তা থেকে আসতে পারে না — এটি সম্পূর্ণভাবে mask থেকে আসে। ঠিক এই কারণেই production structured-output API-গুলো (OpenAI-এর JSON mode / structured outputs, Anthropic-এর tool use, Outlines বা llama.cpp-এর GBNF grammars-এর মতো grammar library) একটি ভালোভাবে trained model-এর নিজে থেকে মেনে চলার উপর নির্ভর না করে **decoding layer**-এ constraint implement করে।

## 5. Function calling: request, parse, execute, respond

Constrained decoding নিশ্চয়তা দেয় যে একটি response *parse* হবে — model-কে আসলে কী *করতে* দেওয়া উচিত সে সম্পর্কে এটি কিছুই বলে না। Function calling হলো সেই convention যা এই ফাঁক বন্ধ করে। Model একটি structured request emit করে যাতে একটি function ও তার argument-এর নাম থাকে:

```json
{"name": "convert_units", "arguments": {"value": 42, "from_unit": "km", "to_unit": "miles"}}
```

বাইরের code — কখনোই model নিজে নয় — সেই JSON parse করে, অনুরোধ করা function-টি আছে কিনা যাচাই করে, **প্রকৃত** parsed argument দিয়ে **প্রকৃত** implementation-এ dispatch করে, execute করে, এবং প্রকৃত return value-টি চূড়ান্ত response-এ জুড়ে দেয়। `example.py` Part B দুটি প্রকৃত Python function-এর (একটি calculator এবং একটি unit-converter) বিরুদ্ধে শুরু থেকে শেষ পর্যন্ত হুবহু এই loop script করে, এবং প্রকৃত মানগুলো সঠিকভাবে ফেরত এসেছে তা যাচাই করতে দুটি প্রত্যাশিত ফলাফলই স্বাধীনভাবে পুনরায় গণনা করে।

## 6. কেন model-এর কাজ ইচ্ছাকৃতভাবে ছোট

পূর্ণ loop-টি কেবল এই কারণেই কাজ করে যে প্রতিটি পক্ষ সেই অংশটি করে যাতে সে সত্যিই ভালো: model-এর পুরো কাজ হলো একটি syntactically বৈধ request emit করা (যা Section 2-4 দেখিয়েছে model-এর মান থেকে স্বাধীনভাবে কাঠামোগতভাবে *নিশ্চিত* করা যায়) — এটি কখনো নিজে `128 * 37 + 6` গণনা করে না, এটি একটি calculator-কে জিজ্ঞেস করে। Calling code-এর কাজ হলো parse, validate, dispatch এবং প্রকৃত system-এর (একটি database, একটি API, একটি calculator) বিরুদ্ধে execute করা, যেগুলোতে model-এর কোনো সরাসরি access নেই। প্রকৃত পরিণতি আছে এমন যেকোনো কিছুর জন্য function calling-কে বিশ্বাসযোগ্য করে ঠিক এই বিভাজনই: একটি model সাবলীলভাবে hallucinate করতে পারে, কিন্তু একটি tool সত্যিই চলে যাওয়ার পর সেই tool-এর return value hallucinate করতে পারে না।

## 7. ReAct-এর তুলনায় এটি কোথায় বসে

[Lesson 3-এর ReAct loop](../03-Tree-of-Thought-and-ReAct/README.md#3-react-yao-et-al-2022-reason-act-observe-repeat) reasoning ("Thought")-কে tool use ("Act")-এর সাথে মিশিয়ে চলে এবং প্রকৃত tool result ফেরত খাওয়ায় ("Observation") — কিন্তু এটি কখনো নির্দিষ্ট করেনি একটি "Act" step *কীভাবে* একটি প্রকৃত চলমান function-এ পরিণত হয়। এই lesson হলো সেই অনুপস্থিত mechanism: প্রতিটি ReAct "Act" ভেতরে ভেতরে হুবহু Section 5-এর request → parse → execute → respond loop, এবং একটি production ReAct agent সাধারণত constrained decoding-ও (Section 2-4) প্রয়োগ করে, যাতে dispatch-এর চেষ্টা করার আগেই প্রতিটি action request সত্যিই parse হওয়ার নিশ্চয়তা থাকে। Structured output এবং function calling agentic prompting pattern থেকে আলাদা কোনো বিষয় নয় — এগুলো তাদের নিচের ভার-বহনকারী mechanism।

## Video Script Outline

1. অনুপ্রেরণা — শুধু prompting বৈধ output-এর *নিশ্চয়তা* দিতে পারে না; downstream code-এর একটি আশার চেয়ে বেশি কিছু দরকার
2. Constrained decoding: sampling-এর আগে অবৈধ token-কে `-inf`-এ mask করা, model-এর মান থেকে স্বাধীন একটি কঠোর নিশ্চয়তা
3. Grammar-এর state machine, `next_valid_chars`, এবং স্বাধীন validity-checker cross-check-এর walkthrough
4. `example.py` Part A-এর walkthrough — একটি untrained random model, masking-সহ 100% বৈধ বনাম এটি ছাড়া প্রায় কখনোই বৈধ নয়
5. Function calling: `{"name", "arguments"}` format এবং request → parse → execute → respond loop
6. `example.py` Part B-এর walkthrough — প্রকৃত Python function সত্যিই execute হচ্ছে, প্রকৃত ফলাফল জুড়ে দেওয়া হচ্ছে, স্বাধীনভাবে যাচাই করা
7. কেন model-এর কাজ ইচ্ছাকৃতভাবে সংকীর্ণ, এবং কেন সেটিই tool use-কে বিশ্বাসযোগ্য করে
8. Recap — Lesson 3-এর ReAct-এর সাথে যোগসূত্র: এটি প্রতিটি "Act" step-এর নিচের mechanism, এবং এই phase-এর শেষ lesson: এখান থেকে, [Phase 08](../../Phase-08-Evaluation-of-LLMs/README.md) কভার করে এই prompting strategy-গুলোর কোনোটি আসলে কাজ করছে কিনা তা কীভাবে প্রকৃতপক্ষে মাপতে হয়

## Further Reading

- OpenAI, *Function calling and other API updates* / *Structured Outputs* documentation
- Anthropic, *Tool use (function calling)* documentation
- Willard & Louf (2023), *Efficient Guided Generation for Large Language Models* (Outlines library — বাস্তবে regex/CFG-constrained decoding)
- Geng et al. (2023), *Grammar-Constrained Decoding for Structured NLP Tasks Without Finetuning*
- Yao et al. (2022), *ReAct: Synergizing Reasoning and Acting in Language Models* — [Lesson 3](../03-Tree-of-Thought-and-ReAct/README.md) থেকে পুনরায় দেখা, এই lesson-এর mechanism-এর উপর নির্মিত strategy layer
