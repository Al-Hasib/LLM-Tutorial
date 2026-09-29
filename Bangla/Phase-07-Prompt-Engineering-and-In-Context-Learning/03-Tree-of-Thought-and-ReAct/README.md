# Tree-of-Thought এবং ReAct

**Phase:** [Prompt Engineering and In-Context Learning](../README.md) · **Topic folder:** `03-Tree-of-Thought-and-ReAct`

## কেন এটি গুরুত্বপূর্ণ

[Lesson 2](../02-Chain-of-Thought-and-Reasoning-Prompts/README.md) reasoning-কে দেখেছিল **একটি রৈখিক chain** হিসেবে — প্রশ্ন থেকে উত্তর পর্যন্ত ধাপের একটি একক sequence, হয়তো কয়েকবার স্বাধীনভাবে resample করে শেষে majority-vote করা। এই lesson সেই সীমাবদ্ধতা দুটি ভিন্ন দিকে সরিয়ে দেয়। Tree-of-Thought একটি একক chain-কে একটি স্পষ্ট **search tree**-তে পরিণত করে, একটিতে প্রতিশ্রুতিবদ্ধ হওয়ার *আগে* একাধিক আংশিক reasoning path অন্বেষণ ও তুলনা করে — যা তখনই গুরুত্বপূর্ণ যখন শুরুর দিকের একটি ভুল মোড় থেকে ফিরে আসা কঠিন। ReAct একটি ভিন্ন সীমাবদ্ধতা সরায়: এটি model-এর reasoning step-গুলোকে **বাইরের জগৎ স্পর্শ করে এমন action**-এর (একটি tool call, একটি lookup, একটি calculator) সাথে মিশিয়ে চলতে দেয়, শুধু prompt-এ যা আছে তা থেকে reasoning করার বদলে। দুটি ধারণাই [Lesson 5](../05-Structured-Output-and-Function-Calling/README.md)-এর পূর্বশর্ত, যা production system-এ একটি model-এর tool-call request কীভাবে নির্ভরযোগ্যভাবে format ও parse করা হয় তা কভার করে।

## এই lesson যা কভার করে

- Tree-of-Thought (ToT): প্রার্থী "thought"-গুলোর উপর tree search হিসেবে reasoning, যেখানে একটি heuristic evaluator branch বিচার ও prune করে
- কেন naive greedy single-path reasoning প্রমাণযোগ্যভাবে আটকে যেতে পারে, এবং একাধিক প্রার্থী branch জীবিত রাখা কীভাবে তা ঠিক করে
- `example.py` Part A: BFS, ToT-style pruned best-first search, এবং naive greedy — একটি অভিন্ন toy search puzzle-এ মুখোমুখি তুলনা
- ReAct: একটি loop-এ Reasoning, Action এবং Observation মিশিয়ে চলা
- কেন ReAct-এর tool call গুরুত্বপূর্ণ: কিছু তথ্য সত্যিই prompt-এ নেই এবং শুধু reasoning তা তৈরি করতে পারে না
- `example.py` Part B: একটি scripted ReAct-style agent যা এমন একটি প্রশ্নের উত্তর দিতে প্রকৃত tool call করে, যার উত্তর শুধু text থেকে দেওয়া সম্ভব নয়

## 1. Tree-of-Thought (Yao et al., 2023): search হিসেবে reasoning

Chain-of-Thought মধ্যবর্তী ধাপের একটি রৈখিক sequence-এ প্রতিশ্রুতিবদ্ধ হয়। **Tree-of-Thought (ToT)** এর বদলে সমস্যা সমাধানকে দেখে **"thought"-এর একটি tree-র উপর search** হিসেবে, যেখানে প্রতিটি node একটি আংশিক solution state:

1. **Generate**: বর্তমান state থেকে কয়েকটি ভিন্ন পরবর্তী "thought" (প্রার্থী পরবর্তী ধাপ) প্রস্তাব করো — এটিই tree-র branching factor।
2. **Evaluate**: প্রতিটি প্রার্থী thought-কে একটি heuristic দিয়ে score করো (হাতে লেখা একটি নিয়ম, অথবা model-কেই নিজের সম্ভাবনা মূল্যায়ন করতে বলা)।
3. **Prune / select**: এক level গভীরে recurse করার আগে শুধু সবচেয়ে সম্ভাবনাময় প্রার্থীদের রাখো, বাকিগুলো বাদ দাও।
4. **Search**: level-by-level (breadth-first-style) বা path-by-path (depth-first-style, backtracking-সহ) generate-evaluate-prune পুনরাবৃত্তি করো, যতক্ষণ না একটি solution পাওয়া যায় বা search budget শেষ হয়।

এটি classical AI search-এর (BFS, best-first search, beam search) একটি সরাসরি সাধারণীকরণ, game board-এর বদলে একটি reasoning trace-এ প্রয়োগ করা — "evaluator" সেই ভূমিকা পালন করে যা ওই classical algorithm-গুলোতে একটি heuristic function (অথবা game-এ একটি value network) পালন করে।

## 2. কেন naive greedy reasoning ব্যর্থ হয়, এবং ToT কীভাবে তা ঠিক করে

একটি একক Chain-of-Thought path হুবহু একটি **greedy, single-path search**: প্রতিটি ধাপে model একটি continuation-এ প্রতিশ্রুতিবদ্ধ হয় এবং সেই পছন্দে আর কখনো ফিরে যায় না। প্রতিটি যুক্তিসঙ্গত পরবর্তী ধাপ যখন সব বিকল্প খোলা রাখে তখন এটি ভালোই কাজ করে — কিন্তু যেসব task-এ শুরুর দিকের একটি স্থানীয়ভাবে আকর্ষণীয় পছন্দ নিঃশব্দে একটি ভালো চূড়ান্ত উত্তরের একমাত্র পথ বন্ধ করে দিতে পারে, সেখানে greedy search-এর ফিরে আসার কোনো উপায় নেই: এটি প্রতিশ্রুতিবদ্ধ হয়ে গেছে, এবং এটি backtrack করে না।

`example.py` Part A একটি প্রকৃত search puzzle দিয়ে এটিকে মূর্ত ও পরিমাপযোগ্য করে: একটি integer থেকে শুরু করে, একটি স্থির operation set (`+3`, `-1`, `*2`) ব্যবহার করে যত কম ধাপে সম্ভব একটি target integer-এ পৌঁছাও। অভিন্ন instance-এ তিনটি প্রকৃত strategy চালানো হয়:

- **BFS oracle** — exhaustive, deduplicated breadth-first search; প্রকৃত optimal (সবচেয়ে কম ধাপের) solution-এর ground truth।
- **ToT-style pruned best-first search** — প্রতিটি depth-এ বর্তমান frontier-এর সব child তৈরি করা হয় (raw branching factor 3, তাই একটি *unpruned* tree `3^depth` হারে বাড়ত — প্রকৃত branch explosion), প্রতিটিকে একটি heuristic (target থেকে দূরত্ব) দিয়ে score করা হয়, এবং শুধু সেরা `beam_width` টি পরের depth-এ টিকে থাকে। এটি হুবহু Yao et al.-এর generate → evaluate → prune loop।
- **Naive greedy** — একটি path, কোনো branching নেই, কোনো backtracking নেই: সবসময় যে একক পরবর্তী ধাপ স্থানীয়ভাবে সেরা দেখায় সেটিই নাও।

200 টি random puzzle instance জুড়ে, ToT-style pruned search বেশিরভাগ সময়েই প্রকৃত optimal solution length-এর সাথে মিলে যায়, অথচ একটি পূর্ণ unpruned tree বা exhaustive BFS-এর যত node লাগত তার একটি ছোট ভগ্নাংশ মাত্র পরীক্ষা করে — অন্যদিকে naive greedy **অর্ধেকেরও বেশি** instance-এ হয় সরাসরি ব্যর্থ হয় (এমন একটি state-এ আটকে যায় যা এটি আগেই visit করেছে, অন্য কিছু চেষ্টা করার কোনো উপায় ছাড়া) অথবা কঠোরভাবে দীর্ঘতর, খারাপ একটি solution-এ পৌঁছায়। `example.py` নিজের run থেকে সঠিক শতাংশগুলো print করে, সাথে একটি মূর্ত instance যেখানে greedy-র একক স্থানীয়-সেরা পছন্দ তাকে এমন একটি solution-এ নিয়ে যায় যা ToT-style search-এর পাওয়া solution-এর চেয়ে 50% বেশি ধাপ নেয়।

## 3. ReAct (Yao et al., 2022): Reason, Act, Observe, পুনরাবৃত্তি

এখন পর্যন্ত প্রতিটি prompting কৌশল শুধু context window-এ ইতিমধ্যে যা লেখা আছে তা থেকে reasoning করে। কিন্তু অনেক বাস্তব প্রশ্নে এমন তথ্য দরকার যা **prompt-এ নেই এবং reasoning করে অস্তিত্বে আনা যায় না** — আজকের তারিখ, একটি database lookup, একটি live API result, এমন একটি arithmetic result যা একটি language model-এর মানসিক গণিতের উপর ভরসা করার পক্ষে খুব বড়। ReAct একটি স্পষ্ট loop দিয়ে এর সমাধান করে:

```
Thought: <what do I need to find out, and why>
Action: <call one specific tool with specific arguments>
Observation: <the tool's real return value, inserted into the context>
Thought: <given that observation, what's next>
...
Final Answer: <once nothing more is needed>
```

প্রতিটি `Observation` প্রকৃতপক্ষে নতুন তথ্য যা tool call করার আগে model-এর কাছে ছিল না — এটি প্রকৃত code (একটি search API, একটি calculator, একটি database query) execute করা থেকে আসে এবং ফলাফলটি context-এ ফেরত দেওয়া হয় যাতে *পরবর্তী* Thought এর উপর condition করতে পারে। এটি সেই একই "reasoning-কে প্রকৃত action-এর সাথে মিশিয়ে চলা" pattern যা প্রতিটি আধুনিক tool-using LLM agent-এর ভিত্তি।

## 4. `example.py` Part B: প্রকৃত tool call-সহ একটি scripted ReAct loop

এই file-এর ReAct অংশে কোনো language model নেই — "Thought" string-গুলো scripted Python, স্পষ্টভাবে লিখে রাখা যাতে *interaction pattern* পুরোপুরি দৃশ্যমান হয়। যা প্রকৃতপক্ষে বাস্তব, scripted নয়, তা হলো tool execution: `lookup_capital`, `lookup_population`, এবং `calculator` প্রকৃত Python function যা এমন প্রকৃত মান ফেরত দেয় যা calling code আগে থেকে জানত না। Demo task — "একটি দেশের রাজধানীর জনসংখ্যা, 1000 দিয়ে ভাগ করে, rounded, কত?" — প্রমাণযোগ্যভাবে শুধু প্রশ্নের text থেকে উত্তর দেওয়া যায় না; এর জন্য দুটি chained lookup এবং একটি প্রকৃত arithmetic গণনা লাগে, এবং script-টি প্রতিটি Thought/Action/Observation ধাপ print করে, সাথে একটি স্বাধীন যাচাই যে agent-এর চূড়ান্ত উত্তর একই পরিমাণ সরাসরি গণনার সাথে মেলে। Lookup entry নেই এমন একটি দেশের বিরুদ্ধে দ্বিতীয় run দেখায় যে loop একটি প্রকৃত "I can't proceed" observation-এ পরিচ্ছন্নভাবে শেষ হয়, একটি উত্তর hallucinate করার বদলে।

## Video Script Outline

1. অনুপ্রেরণা — একটি reasoning chain বনাম chain-এর একটি searchable tree; শুধু-reasoning বনাম জগৎ-স্পর্শ-করতে-পারা reasoning
2. Tree-of-Thought (Yao et al. 2023): generate, evaluate, prune, পুনরাবৃত্তি — classical-search-এর উপমা
3. কেন greedy single-path reasoning প্রমাণযোগ্যভাবে আটকে যেতে পারে: একটি মূর্ত puzzle instance-এর walkthrough
4. `example.py` Part A-এর walkthrough: BFS oracle বনাম ToT-style pruned search বনাম naive greedy, 200-instance batch run থেকে live সংখ্যা
5. ReAct (Yao et al. 2022): Thought / Action / Observation loop, এবং কেন কিছু উত্তরের জন্য প্রকৃত tool call লাগে
6. `example.py` Part B-এর walkthrough: scripted agent-এর পূর্ণ trace, দুটি chained tool call, এবং failure-path run
7. Recap: সাধারণ Chain-of-Thought-এর উপর দুটি স্বাধীন upgrade হিসেবে search breadth (ToT) এবং বাস্তব-জগতের grounding (ReAct)
8. Preview: production-এ একটি model-এর tool-call *request* কীভাবে নির্ভরযোগ্যভাবে format, parse ও validate করা হয় (Lesson 5)

## Further Reading

- Yao et al. (2023), *Tree of Thoughts: Deliberate Problem Solving with Large Language Models*
- Yao et al. (2022), *ReAct: Synergizing Reasoning and Acting in Language Models*
- Wei et al. (2022), *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models* ([Lesson 2](../02-Chain-of-Thought-and-Reasoning-Prompts/README.md) থেকে পুনরায় দেখা)
- Wang et al. (2022), *Self-Consistency Improves Chain of Thought Reasoning in Language Models* ([Lesson 2](../02-Chain-of-Thought-and-Reasoning-Prompts/README.md) থেকে পুনরায় দেখা; একাধিক reasoning path অন্বেষণের আরেকটি উপায়)
- Schick et al. (2023), *Toolformer: Language Models Can Teach Themselves to Use Tools* (একটি model যা scripted বা prompt করা না হয়ে সরাসরি ReAct-style tool use শেখে)
