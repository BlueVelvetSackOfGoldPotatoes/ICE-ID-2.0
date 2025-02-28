import pandas as pd

ultimate_columns = ["ID", "Manntal", "Nafn", "Fornafn", "Millinafn", "Eftirnafn", "Aettarnafn", "Faedingarar", "Kyn",
                    "Stada", "Hjuskapur", "bi_einstaklingur", "bi_baer", "bi_hreppur", "bi_sokn", "bi_sysla",
                    "cleaned_status", "uniqueness_score", "id_individual", "score"]
ultimate_ban_list = ["ID", "Millinafn", "Aettarnafn", "bi_hreppur", "bi_sokn", "bi_sysla", "uniqueness_score",
                     "id_individual", "score"]


class Truth:

    def __init__(self, f, c):
        self.f = f
        self.c = max(min(c, 0.99), 0.01)

    @property
    def wp(self):
        return self.f * self.c / (1 - self.c)

    @property
    def wn(self):
        return (1 - self.f) * self.c / (1 - self.c)

    def revise(self, truth):  # not in-place
        wp = self.wp + truth.wp
        wn = self.wn + truth.wn
        f = wp / (wp + wn)
        c = (wp + wn) / (wp + wn + 1)
        return Truth(f, c)

    @property
    def e(self):
        return self.c * (self.f - 0.5) + 0.5


def preprocessing_ultimate(row_1, row_2):
    # turn two rows into a bunch of formal representations
    ret = set()
    label = -1
    for i in range(min(len(row_1), len(ultimate_columns))):  # Ensure index doesn't go out of bounds
        if ultimate_columns[i] == "Manntal":
            if not pd.isna(row_1[i]) and not pd.isna(row_2[i]):
                dif = str(abs(float(row_1[i]) - float(row_2[i])))
                ret.add("differ_in_" + dif + "_years")
        elif ultimate_columns[i] == "Nafn":
            if not pd.isna(row_1[i]) and not pd.isna(row_2[i]):
                if row_1[i] == row_2[i]:
                    ret.add("same_name")
                else:
                    ret.add("different_name")
        elif ultimate_columns[i] == "Fornafn":
            if not pd.isna(row_1[i]) and not pd.isna(row_2[i]):
                if row_1[i] == row_2[i]:
                    ret.add("same_first_name")
                else:
                    ret.add("different_first_name")
        elif ultimate_columns[i] == "Eftirnafn":
            if not pd.isna(row_1[i]) and not pd.isna(row_2[i]):
                if row_1[i] == row_2[i]:
                    ret.add("same_last_name")
                else:
                    ret.add("different_last_name")
        elif ultimate_columns[i] == "Faedingarar":
            if not pd.isna(row_1[i]) and not pd.isna(row_2[i]):
                if row_1[i] == row_2[i]:
                    ret.add("same_birth_year")
                else:
                    ret.add("different_birth_year")
        elif ultimate_columns[i] == "Kyn":
            if not pd.isna(row_1[i]) and not pd.isna(row_2[i]):
                if row_1[i] == row_2[i]:
                    ret.add("same_gender")
                else:
                    ret.add("different_gender")
        elif ultimate_columns[i] == "Stada":
            if not pd.isna(row_1[i]):
                ret.add("stada_is_" + str(row_1[i]))  # Convert to string
            if not pd.isna(row_2[i]):
                ret.add("stada_is_" + str(row_2[i]))  # Convert to string
        elif ultimate_columns[i] == "Hjuskapur":
            if not pd.isna(row_1[i]) and not pd.isna(row_2[i]):
                if row_1[i] == row_2[i]:
                    ret.add("same_marital_status")
                else:
                    ret.add("different_marital_status")
        elif ultimate_columns[i] == "bi_einstaklingur":
            label = 1 if row_1[i] == row_2[i] else 0
        elif ultimate_columns[i] == "bi_baer":
            if not pd.isna(row_1[i]) and not pd.isna(row_2[i]):
                if row_1[i] == row_2[i]:
                    ret.add("same_residence_ID")
                else:
                    ret.add("different_residence_ID")
        elif ultimate_columns[i] == "cleaned_status":
            if not pd.isna(row_1[i]):
                ret.add("cleaned_status_is_" + str(row_1[i]))  # Convert to string
            if not pd.isna(row_2[i]):
                ret.add("cleaned_status_is_" + str(row_2[i]))  # Convert to string

    return Pattern(ret, Truth(label, 0.9))

class Pattern:

    def __init__(self, statements, truth):
        self.statements = statements
        self.truth = truth

    def __len__(self):
        return len(self.statements)

    def __hash__(self):
        return sum([hash(each) for each in self.statements])

    @property
    def e(self):
        return self.truth.e

    @property
    def c(self):
        return self.truth.c

    @property
    def f(self):
        return self.truth.f

    def match(self, PTR):
        tmp = self.statements.intersection(PTR.statements)
        self_unmatched = Pattern(self.statements - tmp, self.truth)
        matched = Pattern(tmp, self.truth.revise(PTR.truth))
        PTC_unmatched = Pattern(PTR.statements - tmp, PTR.truth)
        if len(self) > len(PTR):
            return (len(matched) / len(self), self.truth.e), self_unmatched, matched, PTC_unmatched
        elif len(self) < len(PTR):
            return (len(matched) / len(PTR), PTR.truth.e), self_unmatched, matched, PTC_unmatched
        else:
            return (len(matched) / len(self), max(PTR.truth.e, self.truth.e)), self_unmatched, matched, PTC_unmatched

class Pattern_pool:
    def __init__(self, pattern_pool_size):
        self.pattern_pool_size = pattern_pool_size
        self.pattern_pool = []

    def add(self, pattern):  # sorted by e
        found_index = None
        
        # Search for a matching pattern based on its statements
        for i, each in enumerate(self.pattern_pool):
            if each.statements == pattern.statements:  # Compare by content (statements)
                found_index = i
                break

        # If we found a similar pattern, replace it if the new one has higher confidence
        if found_index is not None:  # Ensure that found_index is valid
            existing_pattern = self.pattern_pool[found_index]
            if pattern.c > existing_pattern.c:
                self.pattern_pool[found_index] = pattern
        else:
            # Add the pattern to the pool (if not found)
            added = False
            for i in range(len(self.pattern_pool)):
                if pattern.e > self.pattern_pool[i].e:
                    self.pattern_pool.insert(i, pattern)  # Insert pattern in sorted order
                    added = True
                    break

            if not added:
                self.pattern_pool.append(pattern)  # Append at the end if not added yet

            if len(self.pattern_pool) > self.pattern_pool_size:
                self.pattern_pool.pop(len(self.pattern_pool) // 2)  # Remove the middle element

    def get_PTRs(self, num_PTRs):
        # Return a set of top PTRs from both ends of the pattern pool
        return set(self.pattern_pool[:num_PTRs // 2] + self.pattern_pool[-num_PTRs // 2:])

def match_ultimate(row_1, row_2, pattern_pool, num_PTRs, just_eval=False):
    PTC = preprocessing_ultimate(row_1, row_2)
    expectations = []

    # Compare with PTRs from the pattern pool
    for each in pattern_pool.get_PTRs(num_PTRs):
        (sim, e), self_unmatched, matched, PTC_unmatched = PTC.match(each)
        expectations.append(Truth(e, sim))

        if not just_eval:
            # Only add patterns during training (just_eval is False)
            if self_unmatched.statements:
                pattern_pool.add(self_unmatched)
            if matched.statements:
                pattern_pool.add(matched)
            if PTC_unmatched.statements:
                pattern_pool.add(PTC_unmatched)

    if expectations:
        eva = expectations[0]
        for each in expectations[1:]:
            eva = eva.revise(each)
        return eva.e
    else:
        if not just_eval:
            pattern_pool.add(PTC)
        return 0.5

