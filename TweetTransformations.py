from llama_index.core import Settings, SimpleDirectoryReader ,VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.tools import QueryEngineTool
import json
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import AgentStream, ToolCallResult

llm = Ollama(
    model= "gemma3:12b",
    request_timeout=120.0,
    context_window=8128,
    temperature=0.0
)

Settings.llm = llm
Settings.chunk_size=512
Settings.chunk_overlap=64

embed_model = OllamaEmbedding(
    model_name="snowflake-arctic-embed2:latest",
    ollama_additional_kwargs={"mirostat": 0},
)
Settings.embed_model = embed_model

def deconstructTwitterQueryResponse(twitter_response):
    list_of_tweets = []
    for tweet in twitter_response.data:
        tweet_id = tweet.id
        original_tweet = tweet.text
        answer_dict = {
                "original_tweet" : original_tweet,
                "tweet_id": tweet_id
            }
        list_of_tweets.append(answer_dict)
    return list_of_tweets
        
async def translate_to_english(original_tweet):
    response =  await llm.acomplete(f"This tweet is in german, translate it into english. Do not include any other words than the tweet: {original_tweet}")
    return response.text
async def translate_to_german(answer_tweet: str):
    response =  await llm.acomplete(f"This tweet is in english, translate it into german. Do not include any other words than the tweet: {answer_tweet}")
    return response.text

async def checkClaims(tweetlist: list):
    storage_context = StorageContext.from_defaults(persist_dir="./VectorStorage")
    index = load_index_from_storage(storage_context=storage_context)
    query_engine = index.as_query_engine(llm=Settings.llm,similarity_tok_k=5)
    query_engine_tool = QueryEngineTool.from_defaults(query_engine=query_engine, name="RAG_Lookup_tool", description="Query engine tool to look up a knowledge base of documents regarding climate change")
    
    
    
    list_of_relevant_tweets = []
    for tweet in tweetlist:
        original_tweet = tweet.get("original_tweet")
        translated_original_tweet = tweet.get("translated_original_tweet")
        tweet_id = tweet.get("tweet_id")
        prompt = f"""
        This is a tweet from Twitter:
        "{translated_original_tweet}"
        Workflow: 
        2) Extract the claims in the tweet.
        3) Use the RAG_Lookup_tool to fact check the claims in the tweet
        4) If the tweet contains wrong information, write an answer to the tweet in english, where you correct the wrong claims Be direct and critizise missinformation. If the tweet has no claims or all claims are correct answe with: NO_ACTION_NEEDED
        """
        agent = ReActAgent(tools=[query_engine_tool])
        ctx = Context(agent)
        handler = agent.run(prompt, ctx=ctx)

        async for ev in handler.stream_events():
            # if isinstance(ev, ToolCallResult):
            #     print(f"\nCall {ev.tool_name} with {ev.tool_kwargs}\nReturned: {ev.tool_output}")
            if isinstance(ev, AgentStream):
                print(f"{ev.delta}", end="", flush=True)

        response = await handler
        ro= response.model_dump()
        answer_tweet = ro["response"]["blocks"][0]["text"]
        if answer_tweet != "NO_ACTION_NEEDED":
            answer_dict = {
                "original_tweet" : original_tweet,
                "translated_original_tweet": translated_original_tweet,
                "answer_tweet": answer_tweet,
                "tweet_id": tweet_id
            }
            list_of_relevant_tweets.append(answer_dict)

    return list_of_relevant_tweets



