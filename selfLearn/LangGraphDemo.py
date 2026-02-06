#对langgraph框架的初步认识
from  typing import TypedDict,Annotated
from langgraph.graph.message import add_messages

from hello_agents import search


#1.定义全局变量
class SearchState(TypedDict):
    message: Annotated[list,add_messages]
    """
    list: 基础类型，表示这个字段存储一个列表
add_messages: 这是一个特殊的reducer函数，用于定义当向此字段添加新消息时的行为"""
    user_query: str
    search_query: str
    final_answer: str
    step:str
    search_results: str

    #2.定义工作流节点
    #每个节点都是一个执行具体任务的 Python 函数
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient    #优化提示词

load_dotenv()
#初始化模型LLM_MODEL_ID

llm=ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        temperature=0.7)
#初始化TavilyClient
tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
 )


#下面创建三个核心节点
#1.理解与查询节点
def understand_node(state:SearchState) -> dict:
    """
    理解用户查询并生成搜索查询
    """




    user_message=state["message"][-1].content
    """
    从状态字典的"message"键中获取最新的消息内容
state["message"]是一个消息列表（由于使用了add_messages reducer）
[-1]表示取列表中的最后一个元素（最新的一条消息）
.content获取该消息的内容部分"""

# 全部历史对话
    history_text = "\n".join(
        f"{type(m).__name__}: {m.content}"
        for m in state["message"]
    )

    understand_prompt = f"""下面是到目前为止的对话历史：
{history_text}

请重点分析用户最新一条消息（HumanMessage 那条），并：
1. 简洁总结用户想要了解什么
2. 生成最适合搜索引擎的关键词（中英文均可，要精准）

格式：
理解：[用户需求总结]
搜索词：[最佳搜索关键词]
"""
    response = llm.invoke([SystemMessage(content=understand_prompt)])
    response_text=response.content

    #从模型提取关键词
    search_query=user_message
    if "搜索词：" in response_text:
        search_query=response_text.split("搜索词：")[-1].strip()

    return {
        "user_query":user_message,
        "search_query":search_query,
        "step":"understand",
        "message":[AIMessage(content=f"我将为您搜索，{search_query}")]

    }
"""
该节点通过一个结构化的提示，要求 LLM 同时完成“意图理解”和“关键词生成”两个任务，
并将解析出的专用搜索关键词更新到状态的 search_query 字段中，为下一步的精确搜索做好准备。"""

#2.搜索节点
def tavily_search_node(state:SearchState) -> dict:
    """
    执行搜索查询并返回结果
    """
    search_query=state["search_query"]
    """
    从状态字典的"search_query"键中获取搜索查询
    这是上一步理解节点生成的专用搜索关键词"""
    try:
        print(f"正在搜索：{search_query}")
        response = tavily_client.search(query=search_query,
                                        search_depth="basic",
                                        max_results=3,
                                      include_answer=True)

        #处理和格式化搜索结果
        search_results=""
        if "answer" in response and response["answer"]:
            search_results += f"🔍 概括答案：{response['answer']}\n\n"

        if "results" in response and response["results"]:
            search_results += "📋 搜索结果：\n"
            for i, result in enumerate(response["results"], 1):
                search_results += f"{i}. {result.get('title', '无标题')}\n"
                search_results += f"   链接：{result.get('url', '无链接')}\n"
                search_results += f"   内容：{result.get('content', '无内容')}\n\n"
        else:
            search_results = "未找到相关搜索结果"

        #格式化后的结果
        return {
        "search_results": search_results,
        "step": "searched",
        "messages": [AIMessage(content="✅ 搜索完成！正在整理答案...")]
    }

    except Exception as e:
    # ... (处理错误) ...
        error_message = f"搜索失败：{str(e)}"
        print(error_message)

        return {
    "search_results": f"搜索失败：{e}",
    "step": "search_failed",
    "messages": [AIMessage(content="❌ 搜索遇到问题...")]
}
"""
此节点通过 tavily_client.search 发起真实的 API 调用。它被包裹在 try...except 块中，用于捕获可能的异常。
如果搜索失败，它会更新 step 状态为 "search_failed"，这个状态将被下一个节点用来触发备用方案。"""

#3.答案节点
def generate_answer_node(state: SearchState) -> dict:
    """步骤3：基于搜索结果生成最终答案"""
    if state["step"] == "search_failed":
        # 如果搜索失败，执行回退策略，基于LLM自身知识回答
        fallback_prompt = f"搜索API暂时不可用，请基于您的知识回答用户的问题：\n用户问题：{state['user_query']}"
        response = llm.invoke([SystemMessage(content=fallback_prompt)])
    else:
        # 搜索成功，基于搜索结果生成答案
        answer_prompt = f"""基于以下搜索结果为用户提供完整、准确的答案：
用户问题：{state['user_query']}
搜索结果：\n{state['search_results']}
请综合搜索结果，提供准确、有用的回答..."""
        response = llm.invoke([SystemMessage(content=answer_prompt)])

    return {
        "final_answer": response.content,
        "step": "completed",
        "messages": [AIMessage(content=response.content)]
    }


#4.构建图
from langgraph.graph import  StateGraph,START,END
from langgraph.checkpoint.memory import InMemorySaver

def create_search_assistant():
    workflow=StateGraph(SearchState)
    #添加节点
    workflow.add_node("understand_node",understand_node)
    workflow.add_node("search_node",tavily_search_node)
    workflow.add_node("generate_answer_node",generate_answer_node)
    #添加边
    workflow.add_edge(START,"understand_node")
    workflow.add_edge("understand_node","search_node")
    workflow.add_edge("search_node","generate_answer_node")
    workflow.add_edge("generate_answer_node",END)

    #编译图
    app=workflow.compile(checkpointer=InMemorySaver())
   # app=workflow.compile()
    return app

from pprint import pprint
# 主执行部分 - 添加这部分让程序可以运行
if __name__ == "__main__":
    # 创建助手实例
    assistant_app = create_search_assistant()

    # 示例初始状态  一轮对话，初始化状态
    """ initial_state = {
        "message": [HumanMessage(content="适合学习的agent项目有哪些，要求既有深度又有实战性质？")],
        "user_query": "",
        "search_query": "",
        "final_answer": "",
        "step": "start"
    }"""

    config = {
        "configurable": {
            "thread_id": "demo-thread-1"  # 任意字符串，用来标识这次会话
        }
    }
    # 执行工作流
    print("启动AI搜索助手，输入q退出")
    while True:
        user_input = input("请输入问题: ")
        if user_input.lower() == "q":
            break

        # 每一轮只需要传入本轮用户消息，其他状态会从 checkpoint 里自动恢复并合并
        state = {
            "message": [HumanMessage(content=user_input)],
            "step": "start",
        }

        final_state = assistant_app.invoke(state, config)
        print("\n[调试] 当前状态：")
        #pprint(assistant_app.get_state(config).values)
        # 打印对话记忆条数
        current_state = assistant_app.get_state(config)
        message_count = len(current_state.values.get("message", []))
        print("记忆条数：", message_count)
        print("回答：",final_state["final_answer"])
