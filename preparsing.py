import glob, textract
import re
from dateutil.parser import parse


class Preprocessing(object):
    def __init__(self):
        self.__abbrev = None
        self._load_abbrev()
        self._load_case_documents()
    def _load_abbrev(self):
        self.__abbrev = [{(t.split('=')[0]).strip():(t.split('=')[1]).strip()} for t in open('abbreviation.txt').read().splitlines()]

    def _find(self, string):

        # findall() has been used
        # with valid conditions for urls in string
        regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
        url = re.findall(regex, string)
        return [x[0] for x in url]
    def __url_removal(self, filter_ltxt, find_url):
        for u in find_url:
            try:
                url_index = filter_ltxt.index(u)
                next_index = url_index + 4
                prev_ltxt = filter_ltxt[: url_index]
                next_ltxt = filter_ltxt[next_index: ]
                filter_ltxt = prev_ltxt + next_ltxt
            except Exception as e:
                continue

        return  filter_ltxt

    def map_abb(self, t):
        for item in self.__abbrev:
            for k, v in item.items():
                if k in t.split():
                    lt =t.split()
                    index_t = lt.index(k)
                    lt[index_t] = v
                    t = ' '.join(lt)
        return t

    def _load_case_documents(self):
        preprocessed_ldata = []
        lfiles = glob.glob('justinereveal/*')
        for file_path in lfiles:
            mapped_ltxt = []
            event_hearings = textract.process(file_path, encoding='ascii').decode("utf-8").split('Events and Hearings')[1]
            m_filter_ltxt = [t for t in event_hearings.split('\n') if len(t) != 0]
            find_url = self._find(' '.join(m_filter_ltxt))
            if len(find_url) != 0:
                filter_ltxt = self.__url_removal(m_filter_ltxt, find_url)
                for t in filter_ltxt:
                    t = t.replace('\uf0d7', '')
                    changed_txt = self.map_abb(t)
                    try:
                        matches = parse(changed_txt, fuzzy_with_tokens=True)
                        if len(matches) != 0:
                            mapped_ltxt.append([changed_txt])

                    except Exception as e:
                        mapped_ltxt[-1].append(changed_txt)
                        continue



            preprocessed_ldata.append(mapped_ltxt)


        for i in preprocessed_ldata:
            print(i)





obj = Preprocessing()