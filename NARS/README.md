# ICEid

In deciding whether two rows refer to the same invidual, I 1) create patterns based on the two rows (e.g., whether the name is the same), 2) then give the pattern a NAL truth-value indicating the property of the pattern (whether the pattern points to same individual or different individuals). The patterns are restored in a pattern pool for truth-value accumulation, thus filtered out those useful patterns (with extreme expectation values, since is the value is close to 0.5, which means it does not know).

## Pattern

In the file, we have 15 columns, and 3 of them are currently banned ("ID", "baerid", "maki") since they seems useless. And "bi_einstaklingur" is only used as the label, thus we have 11 columns for generating statements. The details are as below:

-	Manntal (census year): statements can be “differ in 5/10/15 years”, pick only one.
-	Nafn (full name): consider the typo, statements is “they are same/different”.
-	Fornafn (first name): as the full name.
-	Eftirnafn (last name): as the full name.
-	Faedingarar (birthday): as the full name.
-	Kyn (gendar): as the full name.
-	Stada (Status or position in society): statements will be “it is what”
-	Hjuskapur (marital status): statement is “they are same/different”
-	Bi_baer (residence ID): statement is “they are same/different”
-	Thsk_maki (IDK): statement is “they are same/different”
-	Cleaned_status (identifier for Status or position in society): tatements will be “it is what”

Currently, 1) I did not include "the way they are different". In the future, if they are different, they way makes them differnt, e.g., "lack of one character" will be there. 2) And I treat "NaN" as a special value. You can skip them and it is 100% okay.

All of them consist a pattern, attached with a truth-value as (1, 0.9) if those two rows are indeed from the same individual, otherwise (0, 0.9).

Patterns can be matched with each other to 1) generate new patterns, 2) revise the truth-value of patterns. For example, pattern A as {(X, Y), (1, 0.9)}, and pattern B as {(Y, Z), (0, 0.9)}. Then they will let us know 1) Y, which is the intersection is nether a sign for same individual nor different individuals. 2) X as the unique part of pattern A, points to the same individual, as well as Y following the same logic.

Those patterns will be forwarded to the pattern pool, which is a buufer of pattern sorted by the expectation values. If it is overwhelmed, the pattern in the middle will be removed. In evaluating two rows (a pattern), some of the patterns in the pool are picked out as references, to which are patterns with extreme (high/low) patterns. E.g., if 4 patterns are used for reference, they will be the first 2 and the last 2.

## Useage

Simply `$ python main.py` and you can see a lot of printed messages. For messages like `0.8248093135287033 | 1`, the number before "|" is the estimated value while the number after is the label, the closer the better. For messages like `{'differ_in_5_years'} f=1 c=0.9`, the contents in brackets are the pattern used for evaluation, the remaining is the truth-value of this pattern. 

To change some hyperparameters, please see `Config.py`.
