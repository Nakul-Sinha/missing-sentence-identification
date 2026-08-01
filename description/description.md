Meridian Ashes: Missing Sentence Identification in Literary Texts
Overview
Objective: Predict the original missing sentence for each of three gaps.

For every test row, select three original sentences from six labelled candidates. Return their candidate letters in gap_1, gap_2, gap_3 order. For example, D>A>F means candidate D belongs in gap_1, candidate A belongs in gap_2, and candidate F belongs in gap_3.

Meridian Ashes is the fictional name of this benchmark. No source story was recovered from an actual fire. Narrative Bridge Authentication means deciding which candidate is the unmodified sentence that originally connected the text on both sides of a gap. This models a real text-integrity problem: detecting and repairing local word-order corruption in literary archives, OCR restoration pipelines, and document collections.

The data describes the texts as short stories but does not provide normalized subgenre labels or publication years. It also does not document a formal sampling frame for selecting the linked pages. The corpus therefore spans a heterogeneous literary collection and should not be treated as a balanced genre sample or a precisely time-bounded benchmark.

The challenge transformation is synthetic and deterministic. Three sentences are removed from each selected story excerpt. The six candidates contain:

Three original missing sentences.
Three counterfeits created by moving a short phrase inside an original sentence.
One counterfeit for each original, with exactly the same whitespace-separated tokens and the same length as that original.
Word counts and candidate length cannot identify which member of a matched pair is original. A solver must use word order, sentence fluency, the text on both sides of each gap, and a one-to-one assignment across all three gaps.

Evaluation
MeridianScore is the mean row quality and ranges from 0 to 1. Higher is better.

For one row, let the prediction be 
p
=
(
p
1
,
p
2
,
p
3
)
p=(p1,p2,p3) and the hidden answer be 
t
=
(
t
1
,
t
2
,
t
3
)
t=(t1,t2,t3).

Position accuracy is the number of exact gap assignments divided by three. It checks 
p
1
=
t
1
p1=t1, 
p
2
=
t
2
p2=t2, and 
p
3
=
t
3
p3=t3.
Membership quality is the number of hidden letters selected anywhere in the prediction divided by three.
Precedence quality evaluates exactly three pairs: 
(
t
1
,
t
2
)
(t1,t2), 
(
t
1
,
t
3
)
(t1,t3), and 
(
t
2
,
t
3
)
(t2,t3). A pair earns one hit only if both letters occur in the prediction and the first appears before the second. The number of hits is divided by three.
Exact quality is 1 only when the complete ordered prediction equals the hidden answer. Otherwise it is 0.
For example, if the hidden answer is D>A>F and the prediction is D>F>A, the precedence pairs are D before A, D before F, and A before F. The first two are correct and the last is wrong, so precedence quality is 2/3.

The row quality is:

q
r
o
w
=
0.30
∗
p
o
s
i
t
i
o
n
a
c
c
u
r
a
c
y
+
0.15
∗
m
e
m
b
e
r
s
h
i
p
q
u
a
l
i
t
y
+
0.05
∗
p
r
e
c
e
d
e
n
c
e
q
u
a
l
i
t
y
+
0.50
∗
e
x
a
c
t
q
u
a
l
i
t
y
qrow=0.30∗positionaccuracy+0.15∗membershipquality+0.05∗precedencequality+0.50∗exactquality

The final score is:

M
e
r
i
d
i
a
n
S
c
o
r
e
=
m
e
a
n
(
q
r
o
w
 over all test rows
)
MeridianScore=mean(qrow over all test rows)

The exact-match component carries half of the score. A prediction with the right candidates in the wrong order receives no exact-match credit.

A malformed, repeated-letter, missing, non-finite, or out-of-alphabet binding receives 0 row quality. A submission in which every binding is invalid receives score 0. A perfect submission receives score 1.

The grader merges rows on id. Missing rows, duplicate ids, foreign ids, missing or renamed columns, and extra columns raise a clean ValueError.

Dataset
The prepared public data contains 10,925 distinct rows derived from 3,722 eligible source stories.

train.csv contains 8,261 labelled rows from 151 author groups.
test.csv contains 2,664 unlabelled rows from 55 different author groups.
sample_submission.csv contains every test id and a constant format-valid example binding.
An author group is the whitespace-normalized value of the source workbook's author field. Every prepared row derived from stories with the same author value belongs to the same group. All rows from one author group are assigned entirely to train or entirely to test before the author field is removed. Author-group overlap between train and test is exactly zero. Author names and group labels are not public model inputs.

train.csv
id (integer) is an independently randomized opaque 15-digit identifier.
gap_1 (string) contains two original sentences before and two after the first missing sentence.
gap_2 (string) contains two original sentences before and two after the second missing sentence.
gap_3 (string) contains two original sentences before and two after the third missing sentence.
candidates (string) contains six newline-separated records. Each record begins with [A] through [F] followed by one candidate sentence.
binding (string) contains the three original candidate letters in gap_1, gap_2, gap_3 order.
The candidates string has this exact layout:

[A] First candidate sentence.
[B] Second candidate sentence.
[C] Third candidate sentence.
[D] Fourth candidate sentence.
[E] Fifth candidate sentence.
[F] Sixth candidate sentence.
test.csv
id (integer) is an independently randomized opaque 15-digit identifier.
gap_1 (string) contains two original sentences before and two after the first missing sentence.
gap_2 (string) contains two original sentences before and two after the second missing sentence.
gap_3 (string) contains two original sentences before and two after the third missing sentence.
candidates (string) uses the same six-line [A] through [F] layout as the training file.
The hidden binding is absent from test.csv. Author, author group, title, source URL, original row number, and split indicator are also absent.

Submission
Write exactly one file to ./working/submission.csv.

The file must contain exactly these columns:

id (integer) contains every test id exactly once.
binding (string) contains three distinct uppercase letters from A through F, separated by >.
The submission must contain exactly 2,664 data rows. Row order does not matter because grading merges on id.

Example using real test ids:

Code snippet

id,binding  
239107279958705,D>A>F  
794116194461966,B>E>A  
340251000425690,F>C>D  

Requirements
Submit exactly 2,664 predictions and cover every test id exactly once.
Use only letters A through F and do not repeat a letter within one binding.
Preserve gap_1>gap_2>gap_3 order.
Do not add columns or omit, duplicate, or invent ids.
A missing, malformed, repeated-letter, non-finite, or out-of-alphabet binding receives zero row quality.
Treat the candidates as one constrained assignment problem rather than three unrestricted choices.
Use the provided GPU for the primary learned sequence model.
Finish within 90 minutes.
What not to use
Do not use pretrained weights, external corpora, plot summaries, source-story lookup, internet access, or runtime downloads.
Do not retrieve stories from source URLs.
Do not fit vocabularies, normalizers, or learned representations on test text.
Harness-side network and filesystem isolation are required because the grader cannot detect external knowledge by itself.
 