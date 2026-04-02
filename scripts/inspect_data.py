from __future__ import annotations

from talentfit.ingest import load_candidates


def main() -> None:
    res = load_candidates("potential-talents.xlsx")
    df = res.df

    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print("\nconnections value_counts (top):")
    print(df["connections"].value_counts(dropna=False).head(20).to_string())
    print("\nrows with null connections:")
    print(df[df["connections"].isna()][["id", "job_title", "connection_raw"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()

