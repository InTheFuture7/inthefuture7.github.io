---
title: ReAct：Synergizing Reasoning and Acting in Language Models
date: 2025-11-18 15:33:00 +0800
categories:
  - agent
tags:
  - agent
  - react
  - langchain
  - langgraph
math: true
publish: true
---

语言模型在推理（例如链式思维提示）和行动（例如 WebGPT、SayCan、ACT-1）方面变得越来越擅长，但这两个方向一直未能融合。 **如果将这两项基本能力结合起来，会怎么样？**

![](https://react-lm.github.io/files/diagram.png)

## ReAct 工作流

下图为 ReAct 工作流，以及 Act-only、Reason-only 的工作流

![](https://raw.githubusercontent.com/InTheFuture7/attachment/main/202601050940379.png)

|      | ReAct                                                                         | Act-only    | Reason-only |
| ---- | ----------------------------------------------------------------------------- | ----------- | ----------- |
| 特点   | 兼具行动和思考                                                                       | 频繁调用工具，缺乏思考 | 单纯思考        |
| 适用场景 | - **知识密集型任务**：密集的思维链生成，与行动和观察交替进行<br>- **交互式决策任务**：稀疏、异步的思想，提供高层次的规划和反思（不太理解） |             |             |

下面举出两个场景来对比直接回答、Reason-only、Act-only、ReAct的回答效果

> 问题：为什么举出这两个场景？

![](https://raw.githubusercontent.com/InTheFuture7/attachment/main/202601050941769.png)

## 代码实现

langchain、langgraph 中有三个版本实现 ReAct

- `from langchain_classic.agents import create_react_agent`
- `from langgraph.prebuilt import create_react_agent`
- `from langchain.agents import create_agent`

> 因为 langchain 升级到 1.0 后，将之前版本中的函数迁移到 `langchain_classic` 库中，详见：https://reference.langchain.com/python/langgraph/agents/
> `langgraph.prebuilt` 同样也是废弃的库。

版本 1 的实现中，会将所有的 AiMessage、ToolMessage 直接拼接到 humanMessage 后面，比较杂乱。而 langgraph 会根据类型分成三类，一组（ai、tool）一组拼接作为大模型输入。新版本的 `langchain.agent` 从工作流上看更加清晰和简洁。

### 版本1：旧版本 langchain 实现

>以下三个版本的实现中，都需要在 `.env` 中配置参数：TAVILY_API_KEY、LANGCHAIN_TRACING_V2、LANGSMITH_API_KEY、LANGCHAIN_PROJECT、MScope_BASE_URL、MScope_API_KEY

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langsmith import Client

load_dotenv()

client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
# 获取提示词，可以通过 prompt.template 输出检查
# include_model=True: the executor feeds the parsing error back to the agent and retries instead of crashing
prompt = client.pull_prompt("hwchase17/react", include_model=True)

# 创建工具
# 需要在 .env 文件中增加：TAVILY_API_KEY="xxx"
tools = [TavilySearchResults(max_results=3)]

# 创建 agent
llm = ChatOpenAI(base_url=os.getenv("MScope_BASE_URL"), 
                 api_key=os.getenv("MScope_API_KEY"),
                 model="deepseek-ai/DeepSeek-V3.2")
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)

# 调用
agent_executor.invoke({'input': '"what is the hometown of the current Australia open winner?'})

"""
{'input': '"what is the hometown of the current Australia open winner?', 'output': "**  \nThe most recent Australian Open men's singles champion (2025) is Jannik Sinner, whose hometown is **Sexten (Sesto)** in South Tyrol, Northern Italy. The women's singles champion (2025) is Aryna Sabalenka, whose hometown is **Minsk, Belarus**."}
"""
```

上述代码使用的提示词

```
Answer the following questions as best you can. You have access to the following tools: 

{tools} 

Use the following format: 

Question: the input question you must answer 
Thought: you should always think about what to do 
Action: the action to take, should be one of [{tool_names}] 
Action Input: the input to the action 
Observation: the result of the action 
... (this Thought/Action/Action Input/Observation can repeat N times) 
Thought: I now know the final answer 
Final Answer: the final answer to the original input question 

Begin! 

Question: {input} 
Thought:{agent_scratchpad}
```

### 版本2：旧版本 langgraph 实现

```python
import os
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
load_dotenv()

llm = ChatOpenAI(base_url=os.getenv("MScope_BASE_URL"), 
                 api_key=os.getenv("MScope_API_KEY"),
                 model="deepseek-ai/DeepSeek-V3.2")
# 每个查询会返回三个结果，包括：title、url、content、score信息
tools = [TavilySearchResults(max_results=3)]
agent = create_react_agent(model=llm,tools=tools)

# # 可视化 agent
# from IPython.display import Image, display
# # 虚边即条件边
# display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

response = agent.invoke(
    {"messages": [{'role': 'user', 'content': 'what is the hometown of the current Australia open winner?'}]}
    )

for msg in response['messages']:
    print(msg.type)

from rich.pretty import pprint
pprint(response)
```

工作流如下：

1. **HumanMessage**：用户询问当前澳大利亚网球公开赛冠军的家乡
2. **AIMessage**：AI识别需要搜索当前澳网冠军信息，调用搜索工具
3. **ToolMessage**：搜索结果显示2025年澳网男子单打冠军是意大利选手Jannik Sinner，他击败了Alexander Zverev成功卫冕；女子单打冠军是Madison Keys
4. **AIMessage**：AI确定需要进一步查询Jannik Sinner的家乡信息，发起更具体的搜索
5. **ToolMessage**：搜索返回Jannik Sinner的个人背景：
    ...
6. **AIMessage**：AI综合信息，清晰说明：
    - Jannik Sinner是当前澳网男子单打冠军
    - 他的出生地是Innichen/San Candido
    - 他的家乡（成长地）是Sexten/Sesto
    - 两个地点都位于意大利最北部靠近奥地利边境的南蒂罗尔，是具有混合意德文化影响的山区

![](https://raw.githubusercontent.com/InTheFuture7/attachment/main/202601050944839.png)

>langgraph 的虚线边（条件边），表示判断是否执行
>
>可以结合 [langsmith 或 langfuse 关注大模型调用过程](https://github.com/wdkns/modern_genai_bilibili/blob/main/agents/langchain-graph/react_langchain_langgraph.ipynb)

### 版本3：langchain 1.0 版本实现

```python
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

tools = [TavilySearchResults(max_results=3)]
  
model = ChatOpenAI(base_url=os.getenv("MScope_BASE_URL"),
					api_key=os.getenv("MScope_API_KEY"),
					model="deepseek-ai/DeepSeek-V3.2")

agent = create_agent(model, tools=tools)
  
print(agent.invoke({"messages": [{"role": "user", "content": "what is the hometown of the current Australia open winner?"}]}))
```

