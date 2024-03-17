from langchain_community.chat_models import ChatOllama
from langchain_core.messages.base import BaseMessage
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import HumanMessagePromptTemplate

from mulmod.img import get_img_base64
from mulmod.retrieve.retriever import RetrievalResult, Retriever


SYS_MSG = (
    "You are machine learning expert on large language models. You are very good \
in giving precise and concise answer to questions about research papers. For every \
answer you are using provided text excerpts and images from that research paper. \
Based on them you answer to the question."
)

QUESTION_MSG = """Based on the provided text, tables and images from the research paper \
answer this question: {question}
This is the provided text and images: {rel_text}"""


class Rag:
    sys_msg: str = ""
    model: str = "llava"
    msg_history: list[BaseMessage] = []

    def __init__(
        self,
        sys_msg: str = SYS_MSG,
        question_msg: str = QUESTION_MSG,
        model: str = "llava",
    ) -> None:
        self.sys_msg = sys_msg
        self.model = model
        self.question_msg = question_msg

        self.llm = ChatOllama(model=self.model)
        self.msg_history = [SystemMessage(content=self.sys_msg)]

    def answer(self, query: str, retriever: Retriever) -> str:
        retrieved = retriever.retrieve(query, 1.0)

        self.msg_history.append(self.process_retrieval(retrieved, query))

        answer = self.llm.invoke(self.msg_history)

        self.msg_history.append(answer)

        return answer.content

    def process_retrieval(self, retrieval: RetrievalResult, query: str) -> HumanMessage:
        content_parts = []
        texts = []

        for doc, _ in retrieval:
            if "img_path" in doc.metadata:
                image_b64 = get_img_base64(doc.metadata["img_path"])
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{image_b64}",
                    }
                )
            else:
                texts.append(doc.page_content)

        temp = HumanMessagePromptTemplate.from_template(self.question_msg)
        text = temp.format(
            question=query,
            rel_text="\n".join(texts),
        ).content

        content_parts.append(
            {
                "type": "text",
                "text": text,
            }
        )

        return HumanMessage(content=content_parts)
