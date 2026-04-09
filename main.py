#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# 2026/04/06
# タイトル：CrewAIを用いたWeb検索エージェント
# 概要：
# ・エージェントがWebを検索して情報を得ます。
# ・シナリオとして、現総理大臣の名前と生い立ちを調べて回答します。
# ・狙いは、現在利用できるLLMは全て衆院総選挙前のデータで学習された
#   モデルのため、Web検索を行った結果を反映できているかを確認します。
# ----------------------------------------------------------------------

from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun
from datetime import datetime
import logging

LLM_MODEL="ollama/gemma4:26b"
OLLAMA_URL="http://192.168.1.64:11434"

# --- 情報ログ出力
#logging.basicConfig(level=logging.INFO)
#logging.getLogger("crewai").setLevel(logging.INFO)
# --- 警告ログ出力
#logging.basicConfig(level=logging.WARNING)
#logging.getLogger("crewai").setLevel(logging.WARNING)
# --- エラーログ出力
#logging.basicConfig(level=logging.ERROR)
#logging.getLogger("opentelemetry").setLevel(logging.ERROR)

# 検索ツールをCrewAI形式にラップする
class SearchTool(BaseTool):
    name: str = "Search"
    description: str = "最新の情報をインターネットで検索するために使用します。"

    def _run(self, query: str) -> str:
        # LangChainのツールを内部で呼び出す
        #print('-->', query)
        result = DuckDuckGoSearchRun().run(query)
        #print('-->', result)
        return result

def main():
    # 現在時刻を取得
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Ollamaの設定 (モデル名は適宜変更してください)
    ollama = LLM(model=LLM_MODEL, base_url=OLLAMA_URL)

    # 検索ツールの準備
    # インスタンス
    search_tool = SearchTool()

    # エージェントの定義
    researcher = Agent(
        role='リサーチ・アナリスト',
        goal='{topic} について、インターネットから最新情報を調査して報告します。',
        backstory='あなたは優秀な情報分析官です。インターネットから得た情報を整理し、分かりやすく伝えることを得意としています。',
        tools=[search_tool],
        llm=ollama,  # ここでOllamaを指定
        verbose=False
    )

    # タスクの定義
    research_task = Task(
        description=(
            "現在時刻は {current_time} です。"
            "{topic} について最新情報をSearchツールを用いてインターネットから検索してください。"
            "必ず現在時刻を基準に、直近の情報を優先して報告してください。"
        ),
        expected_output="調査トピックに関する3つの主要なポイントを含む日本語のレポート",
        agent=researcher
    )

    # クルー（チーム）の結成と実行
    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        verbose=True
    )

    # 実行
    print("## 調査を開始します...")
    result = crew.kickoff(inputs={
        'topic': '現在の内閣総理大臣の名前と生い立ち',
        'current_time': now,
    })

    print('\n' * 2)
    print('=' * 50)
    print("調査結果")
    print('=' * 50)
    print(result)

if __name__ == "__main__":
    main()
