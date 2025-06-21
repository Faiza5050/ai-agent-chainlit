from agents import Runner, Agent, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig
from openai.types.responses import ResponseTextDeltaEvent

import os
from dotenv import load_dotenv
import chainlit as cl

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client,
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)


backend_agent = Agent(
    name="Backend Developer",
    instructions="""
You are an Expert Backend Developer. You are Responsible for Backend Development Tasks.

DO NOT respond to Frontend Related Queries.
"""
)

frontend_agent = Agent(
    name="Frontend Developer",
    instructions="""
You are an Expert Frontend Developer. You are Responsible for Frontend Development Tasks.
DO NOT respond to Backend Related Queries.
"""
)

web_developer_agent = Agent(
    name="Web Developer",
    instructions="""
You are an Expert Web Developer. You are Responsible for both Frontend and Backend Development Tasks.
""",
handoffs=[frontend_agent, backend_agent]
)


@cl.on_chat_start
async def handle_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Hello! Welcome To FAIZA CHATBOT, How Can I Help You?").send()


@cl.on_message
async def handle_message(message: cl.Message):

    history = cl.user_session.get("history")

    history.append(
        {
            "role": "user",
            "content": message.content
        }
    )

    mes = cl.Message(content=" ")
    await mes.send()

    result = Runner.run_streamed(
        web_developer_agent,
        input=history,
        run_config=config
    )

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await mes.stream_token(event.data.delta)
            

    history.append(
        {
            "role": "assistant",
            "content": result.final_output
        }
    )
    cl.user_session.set("history", history)
