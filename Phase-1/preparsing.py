import glob
import re
from dateutil.parser import parse
import textract
import numpy as np



class Preprocessing(object):
    def __init__(self):
        self.__abbrev = None
        self.sequence_length = 50
        self.batch_size = 64
        self.__abbrev = None
        self._load_abbrev()
        self._load_case_documents()

    def _load_abbrev(self):
        self.__abbrev = [{(t.split('=')[0]).strip(): (t.split('=')[1]).strip()} for t in
                         open('abbreviation.txt').read().splitlines()]

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
                next_ltxt = filter_ltxt[next_index:]
                filter_ltxt = prev_ltxt + next_ltxt
            except Exception as e:
                continue

        return filter_ltxt

    def map_abb(self, t):
        for item in self.__abbrev:
            for k, v in item.items():
                if k in t.split():
                    lt = t.split()
                    index_t = lt.index(k)
                    lt[index_t] = v
                    t = ' '.join(lt)
        return t

    def _load_case_documents(self):
        COUNT = 0
        preprocessed_ldata = []
        lfiles = glob.glob('/home/sindhukumari/PycharmProjects/Justin-Falcon-AI-main/justinereveal/*')
        for file_path in lfiles:
            mapped_ltxt = []
            event_hearings = textract.process(file_path, encoding='ascii').decode("utf-8").split('Events and Hearings')[
                1]
            m_filter_ltxt = [t for t in event_hearings.split('\n') if len(t) != 0]
            find_url = self._find(' '.join(m_filter_ltxt))
            if len(find_url) != 0:
                filter_ltxt = self.__url_removal(m_filter_ltxt, find_url)
                for t in filter_ltxt:
                    t = t.replace('\uf0d7', '')
                    t = t.replace('\x0c', '')
                    changed_txt = self.map_abb(t)
                    try:
                        matches = parse(changed_txt, fuzzy_with_tokens=True)
                        if len(matches) != 0:
                            if len(changed_txt) != 0:
                                mapped_ltxt.append([changed_txt.lower()])
                                COUNT += 1


                    except Exception as e:
                        '''if len(changed_txt) != 0:
                            mapped_ltxt[-1].append(changed_txt.lower())'''
                        continue

            preprocessed_ldata.append(mapped_ltxt)

        ''' ltexts = list()
        for index, i in enumerate(preprocessed_ldata):
            doc = ''
            sub = []
            for j in i:
                sub.append(j[0])
            fsub, ssub = sub[0:-1], sub[1:]
            lzipped = list(zip(fsub, ssub))
            ltexts.append(lzipped)
            for k in lzipped:

                doc += '- - ' + k[0] + '\n' + '  - ' + k[1] + '\n'
                with open(('/home/sindhukumari/PycharmProjects/Justin-Falcon-Predictor/data/document_{sindex}.txt'.format(
                        sindex=str(index + 1))), 'w') as f:
                    f.writelines(doc)'''
        '''count = 0
        for i in preprocessed_ldata:
            for j in i:
                for k in j[0].split():
                    count += 1'''
        events = []
        for i in preprocessed_ldata:
            for j in i:
                events.append(j[0])
        lzipped = list(zip(events[:-1], events[1:]))
        return lzipped


obj = Preprocessing()
