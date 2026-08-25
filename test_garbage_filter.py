from pathlib import Path
import csv


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("test-output")

PRIMARY_RESULTS_PATH = (
    OUTPUT_DIR
    / "primary-filter-results.csv"
)

HUMAN_VALIDATION_PATH = (
    Path("racesort-verbose-human-validation.csv")
)

RESULTS_PATH = (
    OUTPUT_DIR
    / "garbage-filter-test-results.csv"
)


# ============================================================
# CANDIDATE RULES
#
# Each rule says:
#
# FILTER_OUT if:
#
#   multi-bike photo
#   AND relative area < area threshold
#   AND relative sharpness < sharpness threshold
#
# These are experiments only.
# ============================================================

RULES = [
    {
        "name": "A_conservative",
        "max_relative_area": 0.10,
        "max_relative_sharpness": 0.50,
    },
    {
        "name": "B_balanced",
        "max_relative_area": 0.20,
        "max_relative_sharpness": 0.45,
    },
    {
        "name": "C_aggressive",
        "max_relative_area": 0.30,
        "max_relative_sharpness": 0.40,
    },
    {
        "name": "D_area_only_small",
        "max_relative_area": 0.12,
        "max_relative_sharpness": 1.01,
    },
]


# ============================================================
# HELPERS
# ============================================================

def load_csv(path):
    with open(
        path,
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        return list(
            csv.DictReader(file)
        )


def normalize_path(value):
    return (
        str(value)
        .replace("\\", "/")
        .strip()
    )


def human_should_process(
    human_role,
):
    """
    Human says we should spend processing time on:
        PRIMARY
        SECONDARY

    Human says we should not spend processing time on:
        NON_PRIMARY
        SKIP
    """

    if human_role in {
        "PRIMARY",
        "SECONDARY",
    }:
        return True

    if human_role in {
        "NON_PRIMARY",
        "SKIP",
    }:
        return False

    return None


# ============================================================
# LOAD DATA
# ============================================================

print(
    f"Loading primary metrics: "
    f"{PRIMARY_RESULTS_PATH}"
)

primary_rows = load_csv(
    PRIMARY_RESULTS_PATH
)


print(
    f"Loading human validation: "
    f"{HUMAN_VALIDATION_PATH}"
)

human_rows = load_csv(
    HUMAN_VALIDATION_PATH
)


# ============================================================
# BUILD HUMAN LOOKUP
# ============================================================

human_lookup = {}

for row in human_rows:

    crop = normalize_path(
        row["crop"]
    )

    human_lookup[
        crop
    ] = row


# ============================================================
# JOIN DATA
# ============================================================

joined_rows = []


for row in primary_rows:

    crop = normalize_path(
        row["crop"]
    )

    human = human_lookup.get(
        crop
    )

    if human is None:

        continue

    human_role = (
        human.get(
            "human_vehicle_role",
            ""
        )
        .strip()
        .upper()
    )

    should_process = (
        human_should_process(
            human_role
        )
    )

    if should_process is None:

        continue


    joined_rows.append(
        {
            "photo":
                row["photo"],

            "crop":
                crop,

            "vehicles_in_photo":
                int(
                    row[
                        "vehicles_in_photo"
                    ]
                ),

            "relative_area":
                float(
                    row[
                        "relative_area"
                    ]
                ),

            "relative_sharpness":
                float(
                    row[
                        "relative_sharpness"
                    ]
                ),

            "human_role":
                human_role,

            "human_should_process":
                should_process,
        }
    )


print()

print(
    f"Rows with usable human labels: "
    f"{len(joined_rows)}"
)

print()


# ============================================================
# TEST EACH RULE
# ============================================================

summary_rows = []


for rule in RULES:

    print("=" * 70)

    print(
        f"RULE: {rule['name']}"
    )

    print("=" * 70)


    filtered_rows = []

    correct_filters = []

    false_filters = []


    for row in joined_rows:

        should_filter = (
            row[
                "vehicles_in_photo"
            ] > 1
            and row[
                "relative_area"
            ]
            < rule[
                "max_relative_area"
            ]
            and row[
                "relative_sharpness"
            ]
            < rule[
                "max_relative_sharpness"
            ]
        )


        if not should_filter:

            continue


        filtered_rows.append(
            row
        )


        if (
            row[
                "human_should_process"
            ]
            is False
        ):

            correct_filters.append(
                row
            )

        else:

            false_filters.append(
                row
            )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        f"Filtered total: "
        f"{len(filtered_rows)}"
    )

    print(
        f"Correct filters: "
        f"{len(correct_filters)}"
    )

    print(
        f"FALSE filters: "
        f"{len(false_filters)}"
    )


    if filtered_rows:

        precision = (
            len(correct_filters)
            / len(filtered_rows)
        )

    else:

        precision = 0.0


    print(
        f"Filter precision: "
        f"{precision:.3f}"
    )


    # --------------------------------------------------------
    # Show mistakes prominently
    # --------------------------------------------------------

    if false_filters:

        print()

        print(
            "WRONGLY FILTERED HUMAN-APPROVED VEHICLES:"
        )

        for row in false_filters:

            print(
                f"  {row['crop']}"
            )

            print(
                f"    human role: "
                f"{row['human_role']}"
            )

            print(
                f"    relative area: "
                f"{row['relative_area']:.3f}"
            )

            print(
                f"    relative sharpness: "
                f"{row['relative_sharpness']:.3f}"
            )


    print()


    summary_rows.append(
        {
            "rule":
                rule["name"],

            "max_relative_area":
                rule[
                    "max_relative_area"
                ],

            "max_relative_sharpness":
                rule[
                    "max_relative_sharpness"
                ],

            "filtered_total":
                len(filtered_rows),

            "correct_filters":
                len(correct_filters),

            "false_filters":
                len(false_filters),

            "filter_precision":
                precision,
        }
    )


# ============================================================
# SAVE SUMMARY CSV
# ============================================================

with open(
    RESULTS_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as file:

    fieldnames = [
        "rule",
        "max_relative_area",
        "max_relative_sharpness",
        "filtered_total",
        "correct_filters",
        "false_filters",
        "filter_precision",
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    writer.writerows(
        summary_rows
    )


# ============================================================
# RECOMMENDATION
# ============================================================

safe_rules = [
    row
    for row in summary_rows
    if row["false_filters"] == 0
]


print("=" * 70)
print("GARBAGE FILTER TEST COMPLETE")
print("=" * 70)


if safe_rules:

    best_safe = max(
        safe_rules,
        key=lambda row:
            row["correct_filters"],
    )

    print(
        "Best zero-false-filter rule:"
    )

    print(
        f"  {best_safe['rule']}"
    )

    print(
        f"  max relative area: "
        f"{best_safe['max_relative_area']}"
    )

    print(
        f"  max relative sharpness: "
        f"{best_safe['max_relative_sharpness']}"
    )

    print(
        f"  safely filtered: "
        f"{best_safe['correct_filters']}"
    )

else:

    print(
        "No tested rule achieved "
        "zero false filtering."
    )


print()

print(
    f"Results CSV: "
    f"{RESULTS_PATH}"
)