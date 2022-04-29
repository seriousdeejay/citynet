# Wikidump processing

from main import correct_cwd

correct_cwd()
## print(os.getcwd())

# import xml.etree.ElementTree as ET


# def strip_tag_name(t):
#     t = elem.tag
#     idx = k = t.rfind("}")
#     if idx != -1:
#         t = t[idx + 1:]
#     return t


# events = ("start", "end")

# title = None
# for event, elem in ET.iterparse('data/enwiki-20190620-pages-articles-multistream.xml', events=events):
#     tname = strip_tag_name(elem.tag)

#     if event == 'end':
#         if tname == 'title':
#             title = elem.text
#         elif tname == 'text':
#             print(title, elem.text)

#     elem.clear()