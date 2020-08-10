import pandas as pd
import numpy as np
import os
import datetime
import math
from cmd import Cmd
from dateutil.relativedelta import relativedelta
# Константы
columns = {
    'Название': 'name',
    'Тип квартир': 'type',
    'Цена за кв.м., руб./кв.м.': 'price'
}
d = [{
    'Объект': None,
    'Тип квартир': None
}]

df_zk=pd.DataFrame(columns=["id_house",'Объект'])
df_house=pd.DataFrame(columns=["id_zk",'Объект',])
df_main = pd.DataFrame(columns=['Объект', 'Тип квартир'])

indicate = 0
lst_group_columns = ["Средневз. стоимость квартиры, руб.","Площадь, кв.м.","Количество проданных, кв.м."]
dict_id_zk = {}
new_house_zk = False
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
    global df_main,database,dict_id_zk,new_house_zk,df_zk,df_house
    df = pd.read_csv(name, sep=';', encoding='cp1251')
    df = df.astype(object).replace({'—': np.nan, '-': np.nan})
    df.iloc[:, 4:] = df.iloc[:, 4:].apply(pd.to_numeric)
    date = name.split('.')[0].split('/')[-1].split("-")
    date = datetime.datetime(int(date[0]), int(date[1]), int(date[2]))
    if indicate == 0:
        df = df.rename(columns={
            'Название': 'Объект',
            'Цена за кв.м., руб./кв.м.': date,
            # "Количество в остатках, шт.": 'Остаток',
            # "Кол-во квартир по проектным декларациям, шт.": "Количество",
            # "Средневз. стоимость квартиры, руб.": "Среднее",
            # "Площадь, кв.м.": "Площадь"
        })
        df1 = df[['Объект', 'Тип квартир', date]]
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
        #создаем бд если она пустая
        df = df.rename(columns={
            'id': 'id_house',
        })
        if database.empty:
            create_bd_id(df)

        #создаем таблцу с основным id_zk и списком id_house
        df['Тип квартир'] = df['Тип квартир'].fillna("-")
        # df_main=df.copy()
        dif =database.groupby("id_zk").agg({
            'id_house': lambda x: sorted(list(x)),
            "Название":"first"
        }).reset_index().sort_values("id_zk")
        #группируем таблицу по id_house чтобы проставить id по домам и жк
        df1 = df[["id_house", "Название","Тип квартир", "Средневз. стоимость квартиры, руб.", "Площадь, кв.м.",
                       "Количество проданных, кв.м."]]
        df1 = df1.groupby("id_house").agg({
            "Название":"first",
            "Тип квартир": lambda x:len(list(x)),
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
                    row["id_zk"]= item["id_zk"]
                # if len(item["id_house"])==count:
                #     break

        #в случае если появились новые дома или жк мы заполняем их новыми id
        df1 = df1.sort_values(["id_house"])
        count = 1
        for index,row in df1.iterrows():
            if math.isnan(row["id_house"]):
                df1.loc[index,"id_house"] = database['id_house'].max()+count
                count+=1
                new_house_zk = True
        df1 = df1.sort_values(["id_zk"])
        for i in df1["Название"]:
            if i not in dict_id_zk:
                dict_id_zk[i] = len(dict_id_zk)
                new_house_zk = True
        for index,row in df1.iterrows():
            df1.loc[index,"id_zk"] = dict_id_zk.get(row["Название"])
        # df1.drop("id", axis='columns', inplace=True)
        #Если мы проранжировали все, то в случае если появились новые дома или жк нужно перезаполнить БД
        if new_house_zk:
            create_bd_id(df1)

        #Далее мы разбиваем на две таблицы: Дома и ЖК

        df1 = df1.sort_values(["id_house"])
        items = df1.to_dict('records')

        count=0
        for index,row in df.iterrows():
            for item in items:
                if row["Название"] == item["Название"] and item["Тип квартир"]>0:
                    df.loc[index,"id_house"] = item["id_house"]
                    item["Тип квартир"]=item["Тип квартир"]-1
                    if item["Тип квартир"]==0:
                        items.remove(item)
                    count += 1
                    break
        result = df.sort_values(["id_house"]).copy()
        # print(df_main)

        #ОТСЮДА НАЧИНАЙ
        '''Надо кароче разобраться с общим заполнением в базы данных и посмотреть с ценной такую же дичь'''
        df = df.rename(columns={
            'Название': 'Объект',
            'Количество проданных, кв.м.': date,
        })
        df_col = df[["id_house", 'Объект', 'Тип квартир', date]]
        df_col = df_col.astype(object).replace({'—': np.nan, '-': np.nan})
        df_col['Тип квартир'] = df_col['Тип квартир'].fillna("-")

        df_col.iloc[:, 3] = pd.to_numeric(df_col.iloc[:, 3])
        # df_col["id_house"]=df_col["id_house"].astype(int)
        if df_main.empty:
            df_main = df_col.sort_values(["id_house"]).copy()
        elif not df_main.empty:
            add_column_to_bd_sell(df_col)

        #####
        # df1 = create_id_col_table(df1,["id_zk","Название"])
        df0 = df1[["id_house","Название","id_zk","Средневз. стоимость квартиры, руб.","Площадь, кв.м.",
                       "Количество проданных, кв.м."]]
        df2 = df1[["id_zk","Название","Средневз. стоимость квартиры, руб.","Площадь, кв.м.",
                       "Количество проданных, кв.м."]]
        df0 = df0.groupby("id_house").agg({
            "Название":"first",
            "id_zk":"first",
            "Площадь, кв.м.": "sum",
            "Средневз. стоимость квартиры, руб.": "sum",
            "Количество проданных, кв.м.": "sum"
        }).reset_index().sort_values(["id_house","id_zk"])

        df2 = df2.groupby("id_zk").agg({
            "Название":"first",
            "Площадь, кв.м.": "sum",
            "Средневз. стоимость квартиры, руб.": "sum",
            "Количество проданных, кв.м.": "sum"
        }).reset_index().sort_values("id_zk")
        df0["Средневз. цена, руб."]=df0["Средневз. стоимость квартиры, руб."]/df0["Площадь, кв.м."]
        df2["Средневз. цена, руб."]=df2["Средневз. стоимость квартиры, руб."]/df2["Площадь, кв.м."]

        df2 = df2.rename(columns={
            'Название': 'Объект',
            'Количество проданных, кв.м.': date,
        })
        df_col2 = df2[["id_zk", 'Объект', date]]
        df_col2 = df_col2.astype(object).replace({'—': np.nan, '-': np.nan})
        df_col2.iloc[:, 2] = pd.to_numeric(df_col2.iloc[:,2])
        # df_col["id_house"]=df_col["id_house"].astype(int)
        if df_zk.empty:
            df_zk = df_col2.sort_values(["id_zk"]).copy()
        elif not df_zk.empty:
            add_column_to_bd_zk(df_col2)

        df0 = df0.rename(columns={
            'Название': 'Объект',
            'Количество проданных, кв.м.': date,
        })
        df_col1 = df0[["id_house", 'Объект', date]]
        df_col1 = df_col1.astype(object).replace({'—': np.nan, '-': np.nan})
        df_col1.iloc[:, 2] = pd.to_numeric(df_col1.iloc[:, 2])
        # df_col["id_house"]=df_col["id_house"].astype(int)
        if df_house.empty:
            df_house = df_col1.sort_values(["id_house"]).copy()
        elif not df_house.empty:
            add_column_to_bd_house(df_col1)
        try:
            os.mkdir(name.split("/")[0] + "-House")
            os.mkdir(name.split("/")[0] + "-ZK")
        except:
            pass
        df0.to_csv(name.split("/")[0]+"-House/"+name.split("/")[-1], index=False, sep=';', encoding='cp1251')
        df2.to_csv(name.split("/")[0]+"-ZK/"+name.split("/")[-1], index=False, sep=';', encoding='cp1251')
        # result.to_csv("1.csv", index=False, sep=';', encoding='cp1251')
        grouped=result.copy()
    return grouped

#Создание БД с Id дома и жк
def create_bd_id(df):
    global database
    df = df.astype(object).replace({'—': np.nan, '-': np.nan})
    df.iloc[:, 4:] = df.iloc[:, 4:].apply(pd.to_numeric)
    # df_main = df.copy()
    if "id_zk" not in df.columns.tolist():
        df = create_id_col_table(df,["id_zk", "Название"])
    df1 = df[["id_house", "Название", "id_zk"]]
    df1 = df1.groupby("id_house").agg({
        "Название": "first",
        "id_zk": "first",
}).reset_index().sort_values(["id_house", "id_zk"])
    if database.empty:
        df1.to_csv("bd.csv", index=False, sep=';', encoding='cp1251')
        database = df1.copy()
    else:
        df2 = pd.read_csv("bd.csv", sep=';', encoding='cp1251')
        result = pd.merge(df2, df1, how='outer', on=["id_house", "Название", "id_zk"])
        result = result.sort_values(["id_house"])
        result.to_csv("bd.csv", index=False, sep=';', encoding='cp1251')
        database = result.copy()

def creat_df_id(lst_col):
    lst_col = ["id_house","id_zk", "Название"]
    global df_main
    df_main = df_main[lst_col+lst_group_columns]
    df_main = df_main.groupby(lst_col[0]).agg({
        lst_col[1:]: "first",
        lst_group_columns:"sum"
        # "Площадь, кв.м.": "sum",
        # "Средневз. стоимость квар9тиры, руб.": "sum",
        # "Количество проданных, кв.м.": "sum"
    }).reset_index().sort_values(["id_house", "id_zk"])

def create_id_col_table(df,lst_col):
    global dict_id_zk
    dict_id_zk = {}
    count =0
    for i in df[lst_col[-1]]:
        if i not in dict_id_zk:
            dict_id_zk[i] = count
            count += 1
    df[lst_col[0]] = [dict_id_zk.get(i) for i in df[lst_col[-1]]]
    return df

def database_to_csv(df,name):
    df.to_csv(name, index=False, sep=';', encoding='cp1251')

# Проверка на
def check_columns(df):
    global df_main,df_zk,df_house
    lst_index = list(df_main)
    if list(df)[-1] in lst_index:
        return ""
    else:
        if isinstance(lst_index[-1], datetime.datetime) and (list(df)[-1].month>=lst_index[-1].month):
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
    if indicate==0:
        data = df_main.loc[:, :"Тип квартир"].copy()
        df_main.drop(["Объект", "Тип квартир"], axis="columns", inplace=True)
        df_main.columns = pd.Series(df_main.columns).apply(lambda x: x.to_period('M').to_timestamp())
        df_main = df_main.groupby(df_main.columns, axis=1).mean()
        is_nan(df_main)
    elif indicate==2:
        data = df_main.loc[:, :"Тип квартир"].copy()
        df_main.drop(["id_house","Объект", "Тип квартир"], axis="columns", inplace=True)
        df_main.columns = pd.Series(df_main.columns).apply(lambda x: x.to_period('M').to_timestamp())
        df_main = df_main.groupby(df_main.columns, axis=1).mean()
    df_main = data.join(df_main)

#Усреднение значений помесячно только для дома и жк
def average_columns_clone(df):
    data = df.loc[:, :"Объект"].copy()
    df.drop(df.columns.tolist()[:2], axis="columns", inplace=True)
    df.columns = pd.Series(df.columns).apply(lambda x: x.to_period('M').to_timestamp())
    df = df.groupby(df.columns, axis=1).mean()
    df = data.join(df)
    return df

#Добавление в бд по ПРОДАЖАМ
def add_column_to_bd_sell(df):
    global df_main
    # df=df.groupby(df.columns).reset_index().reindex(columns=df.columns)
    items=df.to_dict('records')
    # print(df_main[df_main["Объект"]=="25 ЛЕТ ОКТЯБРЯ 14.1"])
    df_main[df.columns[-1]]=np.nan
    col=df.columns.tolist()
    # print(df_main)
    for index, row in df_main.iterrows():
        # print(row["Объект"],index)
        for item in items:
            if row["Объект"] == item["Объект"] and row["Тип квартир"] == item["Тип квартир"] and row["id_house"]==item["id_house"]:
                df_main.loc[index, df.columns[-1]] = item[df.columns[-1]]
                items.remove(item)
                break
    df = pd.DataFrame(items)
    if not df.empty:
        result = pd.merge(df_main, df, how='outer', on=col)
        df_main = result.copy()
    df_main=df_main.sort_values(["id_house"])
    # write_to_excel(df_main, "База по продажам.xlsx")
    # for item in items:
    #     print(item)
    #     df_main.loc[len(df_main),] = item

#Добавление бд ЖК
def add_column_to_bd_zk(df):
    global df_zk
    result = pd.merge(df_zk, df, how='outer', on=["id_zk",'Объект'])
    df_zk = result.copy()

#Добавление бд ДОМА
def add_column_to_bd_house(df):
    global df_house
    result = pd.merge(df_house, df, how='outer', on=["id_house", 'Объект'])
    df_house = result.copy()

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
    global df_main, df_zk, df_house
    if indicate == 0:
        xl = pd.ExcelFile('База по ценам Декарт.xlsx')
        df1 = xl.parse('Цены')
        df_main = change_for_average(df1)
    elif indicate == 2:
        xl = pd.ExcelFile('База по продажам.xlsx')
        df1 = xl.parse('Продажи')
        df_main = df1



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
    if indicate==0:
        df.to_excel(writer, 'Цены', index=False)
    elif indicate==2:
        df.to_excel(writer, 'Продажи', index=False)
    writer.save()


# Расчет проданных кв м
# перебор файлов в папке
def pars_folder_csv(name):
    global df_main,indicate,new_house_zk,df_zk,df_house
    try:
        if indicate == 0 or indicate == 2:
            work_pd_main()
        if len(name.split('.')) == 1:
            dirs = os.listdir(name)
            lst_csv = sorted(dirs, key=lambda x: str(x.split(".")[0]))
            for file in lst_csv:
                if "ok" not in file:
                    new_house_zk=False
                    df1 = create_new_date(name + "/" + file)
                    if indicate == 0:
                        check_columns(df1)
                    elif indicate==1:
                        df1.to_csv(name + "/" + file, index=False, sep=';', encoding='cp1251')
                    elif indicate==2:
                        df1.to_csv(name + "/" + file.split(".")[0]+"-ok.csv", index=False, sep=';', encoding='cp1251')
                        os.remove(name + "/" + file)
                        # df1.to_csv(name + "/" + file, index=False, sep=';', encoding='cp1251')
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
        elif indicate==2:
            average_columns()
            write_to_excel(df_main, "База по продажам.xlsx")
            df_zk=average_columns_clone(df_zk)
            write_to_excel(df_zk, "База по продажам ЖК.xlsx")
            df_house = average_columns_clone(df_house)
            write_to_excel(df_house, "База по продажам Дома.xlsx")


    except FileNotFoundError:
        print("Указан несуществующий путь или не найден файл База по ценам Декарт.xlsx")
    # except KeyError:#если ошибка не исправляеться то тогда закомитеть этот except и посмотреть на что прога указывает
    #     print("Возможно файл не соответствует стандарту")
    #     print("Проверьте, что столбцы разделяются ';', а столбец имеет такой формат 'Цена за кв.м., руб./кв.м.'")


# Замена значений NaN на средние значения
def change_for_average(df):
    df = df.astype(object).replace({'—': np.nan, '-': np.nan})
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

    def do_run(self,inp):
        self.do_add_csv('price_csv')
        self.do_calc_sold('price_csv')
        self.do_bd_zk_house('price_csv')

    def help_run(self):
        print("Проходит все этапы добавление цен в общую таблицу до создания двух бд жк и дома")
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