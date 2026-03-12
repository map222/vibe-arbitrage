import io
import pandas as pd
import yaml
import json
from io import StringIO


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
            df.loc[:, col] = df[col].str.replace('–', '-').str.replace('—', '-')

    def normalize_values(series):
        map_dict = {
            'drama': 'best picture - drama',
            'musical or comedy': 'best picture - musical or comedy',
            'best motion picture - drama': 'best picture - drama',
            'best motion picture - musical or comedy': 'best picture - musical or comedy',
            'comedy/musical': 'best picture - musical or comedy',
            'nominee': 'nominated',
            'won': 'winner',
            'precious': 'precious: based on the novel push by sapphire',
            'birdman': 'birdman or (the unexpected virtue of ignorance)',
            'mulholland drive': 'mulholland dr.',
            'master and commander':'master and commander: the far side of the world',
            'once upon a time in hollywood': 'once upon a time...in hollywood',
            'mulholland dr': 'mulholland dr.',
            'tick, tick... boom!': 'tick, tick...boom!',
            'mrs henderson presents': 'mrs. henderson presents',
            'adaptation.': 'adaptation'
        }
        return series.apply(lambda row: map_dict.get(row, row))

    if 'award' in df.columns:
        df.loc[:, 'award'] = normalize_values(df['award'])
    if 'title' in df.columns:
        df.loc[:, 'title'] = normalize_values(df['title'])
    if 'status' in df.columns:
        df.loc[:, 'status'] = normalize_values(df['status'])
    return df



def is_close(col_a, col_b, threshold= 0.05):
  return (1 - (col_a / col_b)).abs() < threshold


def run_prompt(prompt_name, api_key, model, prompts_file="prompts.yaml"):
    """Run a named prompt from prompts.yaml against an LLM API.

    Parameters:
        prompt_name: key matching a 'name' field in prompts.yaml
        api_key: API key for the model's provider
        model: model identifier (used to detect provider via get_provider)
        prompts_file: path to the YAML file containing prompts

    Prompt config fields:
        name: str — unique identifier
        prompt: str — the prompt text
        output_format: 'tsv' | 'text' (default 'text')
        output_columns: optional list of expected column names
        tsv_file: optional path to a .tsv file whose contents are appended to the prompt

    Returns:
        pandas DataFrame if output_format is 'tsv', otherwise raw response string
    """
    with open(prompts_file) as f:
        prompts = yaml.safe_load(f)

    cfg = next((p for p in prompts if p["name"] == prompt_name), None)
    if cfg is None:
        raise ValueError(f"Prompt '{prompt_name}' not found in {prompts_file}")

    content = cfg["prompt"]

    if "tsv_file" in cfg:
        with open(cfg["tsv_file"]) as f:
            tsv_content = f.read()
        content = f"{content}\n\nHere is the data:\n\n{tsv_content}"

    provider = get_provider(model)

    if provider == "Anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": content}],
        )
        response_text = message.content[0].text

    elif provider == "Gemini":
        from google import genai

        client = genai.Client()

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=content,
            )
        response_text = response.text

    elif provider == "OpenAI":
        import openai
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
        )
        response_text = response.choices[0].message.content

    else:
        raise ValueError(f"Unknown provider for model '{model}'")

    tsv = json.loads(response_text.strip('`').strip('json\n') )['tsv']
    output_df = pd.read_csv(StringIO(tsv), sep = '\t')

    # validate here
    expected_cols = cfg.get("output_columns")
    if expected_cols:
        actual_cols = list(output_df.columns)
        missing = [c for c in expected_cols if c not in actual_cols]
        extra = [c for c in actual_cols if c not in expected_cols]
        if missing or extra:
            print(f"Column mismatch — missing: {missing}, extra: {extra}")
        else:
            print("Columns OK")

    return output_df


def get_provider(model):
    if any(k in model for k in ['haiku', 'sonnet', 'opus', 'claude']):
        return 'Anthropic'
    if any(k in model for k in ['thinking', 'gemini', 'flash', 'fast', 'pro']):
        return 'Gemini'
    if any(k in model for k in ['gpt', 'o1', 'o3', 'openai', 'mini']):
        return 'OpenAI'
    return 'Other'
def get_size(model):
    if any(k in model for k in ['haiku', 'flash', 'fast', 'mini']):
        return 0
    if any(k in model for k in ['thinking', 'sonnet', ]):
        return 1
    if any(k in model for k in ['gpt', 'opus']):
        return 2
    return 'Other'
