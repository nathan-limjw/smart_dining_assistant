# helper function to clean up RAG context before giving to LLM

def format_context_for_llm(context_list: list) -> str:
    '''
    Format list of retrieved context chunks into a single string for LLM prompt
    '''
    if not context_list: # if rag find no result
        return "No relevant restuarant information was found."

    formatted_string = "Here is some context on relevant restaurants:\n\n"

    restaurants = {}
    for item in context_list:
        name = item.get("name", "Unknown Restaurant")
        if name not in restaurants: # if new restaurant, creates a new entry
            restaurants[name] = {
                'categories': item.get('categories', ''),
                'city': item.get('city', ''),
                'snippets': []
            }
        # add snippets, avoid duplicates
        if item.get("chunk_text", "") not in restaurants[name]['snippets']:
            restaurants[name]['snippets'].append(item.get("chunk_text", ''))

    for i, (name, data) in enumerate(restaurants.items()):
        formatted_string += f"--- Restaurant {i+1} ---\n"
        formatted_string += f"Name: {name}\n"
        formatted_string += f"City: {data['city']}\n"
        formatted_string += f"Categories: {data['categories']}\n"

        snippets = "\n".join((f"- \"{s}\"" for s in data['snippets']))
        formatted_string += f"Relevant Review Snippets:\n{snippets}\n\n"
    
    return formatted_string