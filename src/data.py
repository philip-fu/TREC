import pandas


def load(path: str) -> pandas.DataFrame:
    rows = []

    with open(path, "r", encoding="latin-1") as data_in:
        for line in data_in:
            fine_category, input = line.split(None, 1)
            coarse_category, _ = fine_category.split(":")

            rows.append({
                "question": input.strip(),
                "fine_category": fine_category,
                "coarse_category": coarse_category
            })

    return pandas.DataFrame(rows)
