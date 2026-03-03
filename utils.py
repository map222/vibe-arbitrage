import pandas as pd


def clean_award_data(input_df):
    """Clean titles:
    - All lower case
    - Removes years from some titles, e.g. "(2021)"
    Expects columns:
      award: containing name of award (e.g. "best picture")
      status: winner or nominated
      year: int of year the movie was released (not award show date)
      title: title of movie
    """
    df = input_df.copy()

    # lowercase and normalize strings
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
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
            'precious': 'precious: based on the novel push by sapphire',
            'birdman': 'birdman or (the unexpected virtue of ignorance)',
            'mulholland drive': 'mulholland dr.',
            'master and commander':'master and commander: the far side of the world'
        }
        return series.apply(lambda row: map_dict.get(row, row))

    if 'award' in df.columns:
        df.loc[:, 'award'] = normalize_values(df['award'])
    if 'status' in df.columns:
        df.loc[:, 'status'] = normalize_values(df['status'])
    return df

def is_close(col_a, col_b, threshold= 0.05):
  return (1 - (col_a / col_b)).abs() < threshold