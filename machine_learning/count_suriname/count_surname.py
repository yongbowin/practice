#!~/anaconda3/bin/python3.6
# encoding: utf-8

"""
@version: 0.0.1
@author: Yongbo Wang
@contact: yongbowin@outlook.com
@file: test_suriname - count_suriname.py
@time: 8/11/18 4:08 AM
@description: 
"""


class CountSurname:
    surname_dict = {}

    # surname includes 2 words
    def specialSurname(self):
        with open("special_surname", "r") as fp:
            s_lines = fp.readlines()
        special_list = []
        for s_line in s_lines:
            special_list.append(s_line.strip())
        return special_list

    def readTxt(self):

        special_list = self.specialSurname()

        with open("surname", "r") as f:
            lines = f.readlines()
        for line in lines:
            if (line.strip())[:2] not in special_list:
                if (line.strip())[0] not in self.surname_dict:
                    self.surname_dict[(line.strip())[0]] = 1
                else:
                    self.surname_dict[(line.strip())[0]] += 1
            else:
                if (line.strip())[:2] not in self.surname_dict:
                    self.surname_dict[(line.strip())[:2]] = 1
                else:
                    self.surname_dict[(line.strip())[:2]] += 1
        return self.surname_dict


if __name__ == "__main__":
    cousurname = CountSurname()
    surname_dict = cousurname.readTxt()
    # count results
    print(surname_dict)
