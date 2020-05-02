import os
import csv
import json
import sys
import datetime
import numpy as np


def read_company_list(list_file):
    core30 = set()
    large70 = set()
    lines = open(list_file).readlines()
    for i in range(1, len(lines)):
        code, group = lines[i].strip().split(',')
        if group == 'CORE30':
            core30.add(code)
        elif group == 'LARGE70':
            large70.add(code)
    return core30, large70


def extract_news(dir, company_set):
    csv.field_size_limit(sys.maxsize)
    years = os.listdir(dir)
    company_news = {company: dict() for company in company_set}
    for year in years:
        year_dir = os.path.join(dir, year)
        if not os.path.isdir(year_dir) or int(year) < 2013:
            continue
        months = os.listdir(year_dir)
        for month in months:
            month_dir = os.path.join(dir, year, month)
            if not os.path.isdir(month_dir):
                continue
            dates = os.listdir(month_dir)
            for date in dates:
                csv_name = os.path.join(month_dir, date, year + '-' + month + '-' + date + '.csv')
                # num_lines = sum(1 for _ in open(csv_name))
                # print(num_lines)
                with open(csv_name) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"')
                    for row in csv_reader:
                        id, _, story_time, take_time, headline, body, product, topic, rics, _, language, _, _, _, _, _, _ = row
                        if language != 'en':
                            continue
                        take_time = datetime.datetime.strptime(take_time, '%m/%d/%Y %H:%M:%S')
                        # if the news happens after 14:00, can only be used the day after
                        if take_time.hour >= 5:
                            take_time_date_str = (take_time + datetime.timedelta(days=1)).date().strftime('%Y-%m-%d')
                        else:
                            take_time_date_str = take_time.date().strftime('%Y-%m-%d')
                        for ric in rics.split(' '):
                            tem = ric.split('.')
                            if len(tem) >= 2:
                                code = tem[0]
                                market = tem[1]
                            else:
                                code = 'None'
                                market = tem[0]
                            if code in company_set and market == 'T':
                                if take_time_date_str not in company_news[code]:
                                    company_news[code][take_time_date_str] = []
                                company_news[code][take_time_date_str].append(
                                    (id, headline, body, topic, rics, take_time.time().strftime('%H')))
                            '''
                            elif market == 'N225' or market.startswith('TOPX'):
                                for company in company_news:
                                    if take_time_date_str not in company_news[company]:
                                        company_news[company][take_time_date_str] = []
                                    company_news[company][take_time_date_str].append(
                                        (id, headline, body, topic, rics, take_time.time().strftime('%H')))
                            '''
    json.dump(company_news, open('small_company_news.json', 'w'), indent=4)


def cal_news_density():
    company_news = json.load(open('company_news.json'))
    news_total = {}
    over_night = 0
    days = set()
    for company in company_news:
        news_total[company] = 0
        for date in company_news[company]:
            days.add(date)
            news_total[company] += len(company_news[company][date])
            for trade in company_news[company][date]:
                if int(trade[-1]) >= 6:
                    over_night += 1
        print('average news per company per day', news_total[company] / float(len(days)))
    print('average news per day', sum(news_total.values()) / float(len(days)) / len(news_total))
    print('average news per hour', sum(news_total.values()) / float(len(days)) / len(news_total) / 18)
    print('average overnight news number', over_night / float(len(days)) / len(news_total))


def extract_bert(filename):
    id_list = json.load(open('id_of_bert.json'))
    i = 0
    bert_dict = {}
    for line in open(filename):
        # each line is a sentence
        bert_term = json.loads(line.strip())
        # the first index 0 indicates the first token 'CLS'
        # the second index 0 indicates the last layer (-1, -2, -3, -4)
        bert_dict[id_list[i]] = bert_term['features'][0]['layers'][0]['values']
        i += 1
    json.dump(bert_dict, open('extracted_bert.json', 'w'))


def extract_news_headline_for_bert(fname):
    body = []
    id = []
    id_set = set()
    companies = json.load(open(fname))
    for company in companies:
        for news_date in companies[company]:
            news_list = companies[company][news_date]
            for news in news_list:
                if news[0] not in id_set:
                    body.append(news[1])
                    id.append(news[0])
                    id_set.add(news[0])
    write = open('text_for_bert.txt', 'w')
    json.dump(id, open('id_of_bert.json', 'w'))
    for text in body:
        write.write(text + '\n')
    write.close()


def probe(dir):
    csv.field_size_limit(sys.maxsize)
    years = os.listdir(dir)
    headlines = []
    for year in years:
        year_dir = os.path.join(dir, year)
        if not os.path.isdir(year_dir):
            continue
        months = os.listdir(year_dir)
        for month in months:
            month_dir = os.path.join(dir, year, month)
            if not os.path.isdir(month_dir):
                continue
            dates = os.listdir(month_dir)
            for date in dates:
                csv_name = os.path.join(month_dir, date, year + '-' + month + '-' + date + '.csv')
                # num_lines = sum(1 for _ in open(csv_name))
                with open(csv_name) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"')
                    for row in csv_reader:
                        if len(row) != 17:
                            continue
                        id, _, _, _, headline, body, product, topic, rics, _, language, _, _, _, _, _, _ = row
                        if language != 'en':
                            continue
                        for ric in rics.split(' '):
                            tem = ric.split('.')
                            if len(tem) >= 2:
                                code = tem[0]
                                market = tem[1]
                            else:
                                code = 'None'
                                market = tem[0]
                            if market == 'T' and len(headline) > 0:
                                headlines.append(headline)
    headline_len = [len(h) for h in headlines]
    print('headline number', len(headlines), 'headline average length', sum(headline_len)/float(len(headlines)), 'max length', max(headline_len), 'min length', min(headline_len))
    # json.dump(list(news_examples)[:100], open('news_examples.json', 'w'))


def extract_corr_json(fname):
    def extract_code(name):
        return name.split(' ')[0]

    companies = {}
    corr_list = []
    with open(fname) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        head = next(csv_reader)
        company_names = [extract_code(name) for name in head[1:] if len(name) > 0]
        for row in csv_reader:
            name = extract_code(row[0])
            if len(name) == 0:
                continue
            if name not in companies:
                companies[name] = {}
            for i in range(len(company_names)):
                companies[name][company_names[i]] = row[i + 1]
                corr_list.append(abs(float(row[i + 1])))
    for threshold in range(40, 100, 5):
        real_threshold = threshold / 100.
        print('threshold', threshold, np.sum(np.array(corr_list, dtype=np.float32) > real_threshold) / float(len(corr_list)))
    json.dump(companies, open('corr_tpx500.json', 'w'), indent=4)
    return companies


if __name__ == '__main__':
    #extract_corr_json('corr_tpx500.csv')
    extract_news_headline_for_bert('tpx100_company_news.json')
