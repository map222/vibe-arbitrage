import pandas as pd


def clean_award_data(df):
    """Clean titles:
    - All lower case
    - Removes years from some titles, e.g. "(2021)"
    Expects columns:
      award: containing name of award (e.g. "best picture")
      status: winner or nominated
      year: int of year the movie was released (not award show date)
      title: title of movie
    """

    # lowercase and normalize strings
    for col in df.columns:
        if df[col].dtype == 'O':
            df.loc[:, col] = df[col].str.lower()
            df.loc[:, col] = df[col].str.replace(r'\s*\(\d{4}\)\s*', '', regex=True).str.strip()
            df.loc[:, col] = df[col].str.normalize('NFKD')

    def normalize_values(series):
        map_dict = {
            'drama': 'best picture - drama',
            'best motion picture - drama': 'best picture - drama',
            'best motion picture - musical or comedy': 'best picture - musical or comedy',
            'comedy/musical': 'best picture - musical or comedy',
            'nominee': 'nominated',
            'won': 'winner',
        }
        return series.apply(lambda row: map_dict.get(row, row))

    if 'category' in df.columns:
        df.loc[:, 'category'] = normalize_values(df['category'])
    if 'status' in df.columns:
        df.loc[:, 'status'] = normalize_values(df['status'])
    return df
