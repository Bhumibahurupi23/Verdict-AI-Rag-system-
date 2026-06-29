def generate_answer(query, docs):

    answer = "### Key Points\n\n"

    for doc in docs[:5]:

        text = doc.page_content.replace("\n", " ")

        if len(text) > 150:
            text = text[:150] + "..."

        answer += f"• {text}\n\n"

    return answer