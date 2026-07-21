from langchain_core.messages import HumanMessage, SystemMessage


class HumanPromptBuilder:
    def build(self) -> HumanMessage:
        content = """
            Project Gutenberg hosts a complete plain-text copy of F. Scott Fitzgerald's
            The Great Gatsby.

            URL: https://www.gutenberg.org/files/64317/64317-0.txt

            Use the available tools to answer the following questions:

            1. Count the number of lines in the complete document that contain the
            substring `Gatsby`. Count matching lines, not occurrences within one line.
            2. Find the 1-based number of the first line in the complete document that
            contains `Daisy`.
            3. Write a neutral synopsis of The Great Gatsby in exactly two sentences.

            Use `fetch_text` to retrieve the document before calculating the line count
            or line number. Use `count_lines_containing` and `find_first_line` to verify
            the numeric answers. Do not estimate or invent numeric values.

            Return the final answer as an `Answer` structured response with these fields:

            - `gatsby_line_count`: the verified line count for `Gatsby`, or `null` when
            it could not be verified.
            - `first_daisy_line_number`: the verified 1-based line number for the first
            `Daisy`, or `null` when it could not be verified.
            - `synopsis`: a neutral synopsis in exactly two sentences, or `null` only
            when a synopsis cannot be provided.
            - `how_you_computed_counts`: the tools used, their relevant results, and the
            reason whenever a numeric field is `null`.
            - `errors`: all tool or execution error messages encountered. Use an empty
            list when no errors occurred.

            Do not return a normal prose answer outside the `Answer` structured response.
        """
        return HumanMessage(content=content)


class SystemPromptBuilder:
    def build(self) -> SystemMessage:
        content = """
            You are a literary data assistant.

            ## Capabilities

            - `fetch_text`: loads document text from a URL into the conversation.
            Do not guess line counts or positions—ground them in tool results from the saved file.
            - `count_lines_containing`: counts the number of lines in a text that contain a given substring.
            - `find_first_line`: finds the 1-based line number of the first line in
        """
        return SystemMessage(content=content)
