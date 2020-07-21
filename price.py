import pandas as pd
import numpy as np
import os
import datetime
import math
from cmd import Cmd
from dateutil.relativedelta import relativedelta

columns = {
    'Название': 'name',
    'Тип квартир': 'type',
    'Цена за кв.м., руб./кв.м.': 'price'
}
d = [{
    'Объект': None,
    'Тип квартир': None
}]
df_main = pd.DataFrame(columns=['Объект', 'Тип квартир'])
indicate = 0
lst_group_columns = ["Средневз. стоимость квартиры, руб.","Площадь, кв.м.","Количество проданных, кв.м."]
dict_id_zk = {}
try:
    database = pd.read_csv("bd.csv", sep=';', encoding='cp1251')
    db = database.groupby("Название").agg({
        "id_zk":"first"
    }).reset_index().sort_values(["id_zk"])
    for index,row in db.iterrows():
        if row["Название"] not in dict_id_zk:
            dict_id_zk[row["Название"]] = row["id_zk"]
except:
    database = pd.DataFrame(columns=["id", "Название", "id_zk"])


# Создание столбца для основного файла
def create_new_date(name):
    global df_main,database,dict_id_zk
    df = pd.read_csv(name, sep=';', encoding='cp1251')
    df = df.astype(object).replace({'—': np.nan, '-': np.nan})
    df.iloc[:, 4:] = df.iloc[:, 4:].apply(pd.to_numeric)
    date = name.split('.')[0].split('/')[-1].split("-")
    date = datetime.datetime(int(date[0]), int(date[1]), int(date[2]))
    if indicate == 0:
        df = df.rename(columns={
            'Название': 'Объект',
            'Цена за кв.м., руб./кв.м.': date,
            "Количество в остатках, шт.": 'Остаток',
            "Кол-во квартир по проектным декларациям, шт.": "Количество",
            "Средневз. стоимость квартиры, руб.": "Среднее",
            "Площадь, кв.м.": "Площадь"
        })
        df1 = df[['Объект', 'Тип квартир', date]]
        df1 = df1.astype(object).replace({0: np.nan, '—': np.nan, '-': np.nan})
        df1.iloc[:, 2] = pd.to_numeric(df1.iloc[:, 2])
        grouped = df1.groupby(['Объект', 'Тип квартир'], sort=False).mean()
    elif indicate == 1:
        # df = df.astype(object).replace({'—': np.nan, '-': np.nan})
        # df.iloc[:, 4:]=df.iloc[:, 4:].apply(pd.to_numeric)
        df["Количество проданных, кв.м."] = (df["Кол-во квартир по проектным декларациям, шт."]
                                  - df["Количество в остатках, шт."]) * df["Площадь, кв.м."]
        df = df.astype(object)
        grouped = df.copy()
    elif indicate == 2:
        df = df.rename(columns={
            'id': 'id_house',
        })
        if database.empty:
            create_bd_id(df)

        df_main=df.copy()
        dif =database.groupby("id_zk").agg({
            'id_house': lambda x: sorted(list(x)),
            "Название":"first"
        }).reset_index().sort_values("id_zk")
        df1 = df_main[["id_house", "Название", "Средневз. стоимость квартиры, руб.", "Площадь, кв.м.",
                       "Количество проданных, кв.м."]]
        df1 = df1.groupby("id_house").agg({
            "Название":"first",
            "Площадь, кв.м.": "sum",
            "Средневз. стоимость квартиры, руб.": "sum",
            "Количество проданных, кв.м.": "sum"
        }).reset_index().sort_values(["id_house"])
        items = dif.to_dict('records')
        df1["id_house"] = np.nan
        df1["id_zk"]=np.nan
        for item in items:
            count = 0
            for index,row in df1.iterrows():
                if row["Название"]==item["Название"] and len(item["id_house"])>count:
                    df1.loc[index,"id_house"] = item["id_house"][count]
                    count +=1
                if row["Название"]==item["Название"]:
                    df1.loc[index, "id_zk"] = item["id_zk"]
        df1 = df1.sort_values(["id_house"])
        count = 0
        for index,row in df1.iterrows():
            if math.isnan(row["id_house"]):
                df1.loc[index,"id_house"] = len(database)+count
                count+=1
        df1 = df1.sort_values(["id_zk"])
        for i in df1["Название"]:
            if i not in dict_id_zk:
                dict_id_zk[i] = len(dict_id_zk)
        for index,row in df1.iterrows():
            df1.loc[index,"id_zk"] = dict_id_zk.get(row["Название"])
        # df1.drop("id", axis='columns', inplace=True)
        create_bd_id(df1)
        result = df1.sort_values(["id_house"]).copy()
        create_id_col_table(["id_zk","Название"])
        df1 = df_main[["id_house","Название","id_zk","Средневз. стоимость квартиры, руб.","Площадь, кв.м.",
                       "Количество проданных, кв.м."]]
        df2 = df_main[["id_zk","Название","Средневз. стоимость квартиры, руб.","Площадь, кв.м.",
                       "Количество проданных, кв.м."]]
        df1 = df1.groupby("id_house").agg({
            "Название":"first",
            "id_zk":"first",
            "Площадь, кв.м.": "sum",
            "Средневз. стоимость квартиры, руб.": "sum",
            "Количество проданных, кв.м.": "sum"
        }).reset_index().sort_values(["id_house","id_zk"])
        df2 = df2.groupby("Название").agg({
            "id_zk":"first",
            "Площадь, кв.м.": "sum",
            "Средневз. стоимость квартиры, руб.": "sum",
            "Количество проданных, кв.м.": "sum"
        }).reset_index().sort_values("id_zk")
        df1["Средневз. цена, руб."]=df1["Средневз. стоимость квартиры, руб."]/df1["Площадь, кв.м."]
        df2["Средневз. цена, руб."]=df2["Средневз. стоимость квартиры, руб."]/df2["Площадь, кв.м."]
        # for index,row in df.iterrows():
        #     print(df["id_ZK"].iloc[index])
        #     df["id_ZK"].iloc[index]=dict_ZK.get(row[2])
        try:
            os.mkdir(name.split("/")[0] + "-House")
            os.mkdir(name.split("/")[0] + "-ZK")
        except:
            pass
        df1.to_csv(name.split("/")[0]+"-House/"+name.split("/")[-1], index=False, sep=';', encoding='cp1251')
        df2.to_csv(name.split("/")[0]+"-ZK/"+name.split("/")[-1], index=False, sep=';', encoding='cp1251')
        # result.to_csv("1.csv", index=False, sep=';', encoding='cp1251')
        grouped=result.copy()
    return grouped

#Создание БД с Id дома и жк
def create_bd_id(df):
    global df_main,database
    df = df.astype(object).replace({'—': np.nan, '-': np.nan})
    df.iloc[:, 4:] = df.iloc[:, 4:].apply(pd.to_numeric)
    df_main = df.copy()
    if "id_zk" not in df_main.columns.tolist():
        create_id_col_table(["id_zk", "Название"])
    df1 = df_main[["id_house", "Название", "id_zk"]]
    df1 = df1.groupby("id_house").agg({
        "Название": "first",
        "id_zk": "first",
}).reset_index().sort_values(["id_house", "id_zk"])
    df1.to_csv("bd.csv", index=False, sep=';', encoding='cp1251')
    database = df1.copy()

def creat_df_id(lst_col):
    lst_col = ["id_house","id_zk", "Название"]
    global df_main
    df_main = df_main[lst_col+lst_group_columns]
    df_main = df_main.groupby(lst_col[0]).agg({
        lst_col[1:]: "first",
        lst_group_columns:"sum"
        # "Площадь, кв.м.": "sum",
        # "Средневз. стоимость квартиры, руб.": "sum",
        # "Количество проданных, кв.м.": "sum"
    }).reset_index().sort_values(["id_house", "id_zk"])

def create_id_col_table(lst_col):
    global df_main,dict_id_zk
    dict_id_zk = {}
    count =0
    for i in df_main[lst_col[-1]]:
        if i not in dict_id_zk:
            dict_id_zk[i] = count
            count += 1
    df_main[lst_col[0]] = [dict_id_zk.get(i) for i in df_main[lst_col[-1]]]

def database_to_csv(df,name):
    df.to_csv(name, index=False, sep=';', encoding='cp1251')

# Проверка на
def check_columns(df):
    global df_main
    lst_index = list(df_main)
    if df_main.empty:
        df_main = df.copy()
    elif list(df)[-1] in lst_index:
        return ""
    else:
        if isinstance(lst_index[-1], datetime.datetime):
            if ((lst_index[-1].month - list(df)[-1].month) == -1) and ((lst_index[-1].year - list(df)[-1].year) == 0):
                add_column_to_main(df)
                # average_columns()
                return df
            elif ((lst_index[-1].month - list(df)[-1].month) == 0) and ((lst_index[-1].year - list(df)[-1].year) == 0):
                return add_column_to_main(df)
            else:
                df_main[lst_index[-1] + relativedelta(months=+1)] = np.nan
                return check_columns(df)
        else:
            return add_column_to_main(df)


# усреднение значений по месячно
def average_columns():
    global df_main
    data = df_main.loc[:, :"Тип квартир"].copy()
    df_main.drop(["Объект", "Тип квартир"], axis="columns", inplace=True)
    df_main.columns = pd.Series(df_main.columns).apply(lambda x: x.to_period('M').to_timestamp())
    df_main = df_main.groupby(df_main.columns, axis=1).mean()
    is_nan(df_main)
    df_main = data.join(df_main)


# Добавление столбца к основному
def add_column_to_main(df):
    global df_main
    result = pd.merge(df_main, df, how='outer', on=['Объект', 'Тип квартир'])
    result = change_for_average(result)
    df_main = result.copy()


# Замена nan-значений
def is_nan(df):
    for index, row in df.iterrows():
        if not math.isnan(row[-1]):
            count = list(row.isnull()).index(False)
            while count < len(row):
                if math.isnan(row[count]):
                    row[count] = row[count - 1] + (row[-1] - row[count - 1]) / (len(row) - count + 1)
                count += 1


# Работа с основной таблицей
def work_pd_main():
    global df_main
    xl = pd.ExcelFile('База по ценам Декарт.xlsx')
    df1 = xl.parse('Цены')
    df_main = change_for_average(df1)


# Парсинг из excel в csv
def excel_to_csv(name):
    if len(name.split('.')) == 1:
        os.mkdir(name + "_csv")
        dirs = os.listdir(name)
        lst_csv = sorted(dirs, key=lambda x: str(x.split(".")[0]))
        for file in lst_csv:
            xl = pd.ExcelFile(name + "/" + file)
            df1 = xl.parse('Цены')
            df1.to_csv(name + "_csv/" + file.split(".")[0] + '.csv', index=False, sep=';', encoding='cp1251')
    else:
        xl = pd.ExcelFile(name)
        df1 = xl.parse('Цены')
        df1.to_csv(name.split('.')[0].split('/')[-1] + '.csv', index=False, sep=';', encoding='cp1251')


# Парсинг из csv в excel
def csv_to_excel(name):
    if len(name.split('.')) == 1:
        os.mkdir(name + "_excel")
        dirs = os.listdir(name)
        lst_csv = sorted(dirs, key=lambda x: str(x.split(".")[0]))
        for file in lst_csv:
            df = pd.read_csv(name + "/" + file, sep=';', encoding='cp1251')
            writer = pd.ExcelWriter(name + "_excel/" + file.split(".")[0] + ".xlsx", engine='xlsxwriter')
            df.to_excel(writer, 'Цены')
            writer.save()
    else:
        writer = pd.ExcelWriter(name.split('.')[0].split('/')[-1] + ".xlsx", engine='xlsxwriter')
        df = pd.read_csv(name, sep=';', encoding='cp1251')
        df.to_excel(writer, 'Цены', index=False)
        writer.save()


# Запись таблицы в Excel
def write_to_excel(df, name):
    writer = pd.ExcelWriter(name)
    df.to_excel(writer, 'Цены', index=False)
    writer.save()


# Расчет проданных кв м
# перебор файлов в папке
def pars_folder_csv(name):
    global df_main
    try:
        if indicate == 0:
            work_pd_main()
        if len(name.split('.')) == 1:
            dirs = os.listdir(name)
            lst_csv = sorted(dirs, key=lambda x: str(x.split(".")[0]))
            for file in lst_csv:
                df1 = create_new_date(name + "/" + file)
                if indicate == 0:
                    check_columns(df1)
                elif indicate==1 or indicate==2:
                    df1.to_csv(name + "/" + file, index=False, sep=';', encoding='cp1251')


        else:
            df1 = create_new_date(name)
            if indicate == 0:
                check_columns(df1)
            elif indicate == 1 or indicate==2:
                df1.to_csv(name, index=False, sep=';', encoding='cp1251')
            # elif indicate==2:
            #     df1.to_csv("Название"+name, index=False, sep=';', encoding='cp1251')
        if indicate == 0:
            average_columns()
            write_to_excel(df_main, "База по ценам Декарт.xlsx")
    except FileNotFoundError:
        print("Указан несуществующий путь или не найден файл База по ценам Декарт.xlsx")
    # except KeyError:
    #     print("Возможно файл не соответствует стандарту")
    #     print("Проверьте, что столбцы разделяются ';', а столбец имеет такой формат 'Цена за кв.м., руб./кв.м.'")


# Замена значений NaN на средние значения
def change_for_average(df):
    df = df.astype(object).replace({0: np.nan, '—': np.nan, '-': np.nan})
    df.iloc[:, 2:].apply(pd.to_numeric, errors='ignore')
    return df


# Класс команд для управления файлом
class MyPrompt(Cmd):
    prompt = 'pb>'
    intro = 'Welcome! Type ? to list commands'

    def do_exit(self, inp):
        print('Bye')
        return True

    def help_exit(self):
        print('exit the application. Shorthand: x q Ctrl-D')

    def do_csv_to_excel(self, inp):
        csv_to_excel(inp)

    def help_csv_to_excel(self):
        print("Перевод значений из csv-файла в Excel")

    def do_excel_to_csv(self, inp):
        excel_to_csv(inp)

    def help_excel_to_csv(self):
        print("Перевод значений из Excel в csv")

    def do_add_csv(self, inp):
        global indicate
        indicate = 0
        pars_folder_csv(inp)

    def help_add_csv(self):
        print("Добавить csv-файл или папку csv файлов к основной таблице")

    def do_calc_sold(self, inp):
        global indicate
        indicate = 1
        pars_folder_csv(inp)

    def help_calc_sold(self):
        print("Добавить в таблицу столбец Проданных кв. м.")

    def do_bd_zk_house(self, inp):
        global indicate
        indicate = 2
        pars_folder_csv(inp)

    def help_bd_zk_house(self):
        print("Разделяет файл на 2 базы данных для Домов и для ЖК")

    # def do_main_df(self, inp):
    #     global df_main
    #     work_pd_main(str(inp))
    #
    # def help_main_df(self):
    #     print("Создать основную таблицу для ее дальнейшего заполнения")

    def default(self, inp):
        pass

# def main():
#     global indicate
#     # indicate = 1
#     # pars_folder_csv("2020-07-11-data.csv")
#     indicate=2
#     pars_folder_csv("csv2")


if __name__ == '__main__':
    MyPrompt().cmdloop()
    # main()