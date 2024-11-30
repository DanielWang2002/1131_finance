import datetime

import pandas as pd

from modules.Fundamental import FundamentalData
from modules.News import GNews
from modules.LLM import GeminiAPI


def mix_output(fd_output, news_output):
    return fd_output + "\n" + "以下是最近的十筆新聞：\n" + news_output


if __name__ == "__main__":

    ### FundamentalData ###
    fd_api = ""

    fd = FundamentalData(api_token=fd_api)

    df = fd.get_fundamental_df(stock_id="2330", start_date="2023-01-01")
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] == fd.get_quarter(datetime.datetime.now())]
    # 將NaN超過 50% 的欄位刪除
    df = df.dropna(thresh=int(len(df) * 0.5), axis=1)

    # 將財務資料的 DataFrame 轉換為自然語言格式，使用變數
    # 過濾重複的 origin_name_x 和 origin_name_y
    # df_unique = df.drop_duplicates(subset=["origin_name_x", "origin_name_y", "origin_name"])
    fd_output = fd.summarize_financial_data(df)
    ### FundamentalData ###

    ### GNews ###
    gnews_api = ""

    gnews = GNews(api_key=gnews_api)

    df = gnews.fetch_news(query="2330", max_results=100)

    news_output = gnews.news_to_natural_language(df)

    ### GNews ###

    ### LLM ###
    stock_info = mix_output(fd_output, news_output)

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    head = f"請透過以下資訊對台積電 (2330) 進行分析，並在最後提供投資建議、理由，以及1~100的投資信心值。今天是{today}。請使用繁體中文進行回覆。不需要給我投資風險的相關警告或是免責聲明。"

    llm_api = ""
    llm = GeminiAPI(llm_api)

    prompt = head + stock_info
    llm_output = llm.generate_content(prompt=prompt)
    print(llm_output)
