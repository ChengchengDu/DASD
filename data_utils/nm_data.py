import os

def deal_Nema_data(path, data_type):
    source_path = os.path.join(path, data_type+"_content")
    query_path = os.path.join(path, data_type+"_query")
    summary_path = os.path.join(path, data_type+"_summary")

    sources = []
    querys = []
    summarys = []

    with open(source_path, 'r', encoding="utf-8") as sf, \
            open(query_path, 'r', encoding='utf-8') as qf, \
            open(summary_path, 'r', encoding="utf-8") as mf:
        source_lines = sf.readlines()
        query_lines = qf.readlines()
        summary_lines = mf.readlines()

        for (source_line, query_line, summary_line) in zip(source_lines, query_lines, summary_lines):
            source_line = " ".join(source_line.split()[1:-1]).strip()
            query_line = " ".join(query_line.split()[1:-1]).strip()
            if query_line.find(":"):
                pos = query_line.find(":")
                query_line = query_line[pos + 1:].strip()
            summary_line = " ".join(summary_line.split()[1:-1]).strip()
            sources.append(source_line + "\n")
            querys.append(query_line + "\n")
            summarys.append(summary_line + "\n")

    source_path = os.path.join(path, data_type + ".source")
    query_path = os.path.join(path, data_type + ".query")
    summary_path = os.path.join(path, data_type + ".summary")

    with open(source_path, 'w', encoding="utf-8") as sf, \
            open(query_path, 'w', encoding='utf-8') as qf, \
            open(summary_path, 'w', encoding="utf-8") as mf:
        sf.writelines(sources)
        qf.writelines(querys)
        mf.writelines(summarys)

if __name__ == "__main__":
    deal_Nema_data("C:\\Users\\admin\\Downloads\\PrekshaNema25\\", data_type="valid")





