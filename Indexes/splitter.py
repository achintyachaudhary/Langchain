from langchain_text_splitters import RecursiveCharacterTextSplitter



def splitter_user_input_file(text):
    splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " "]
    )

    chunks = splitter.split_text(text)
    return chunks