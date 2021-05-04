# Copyright (c) 2021 by Cisco Systems, Inc.
# All rights reserved.
#
# Author: larzhang@cisco.com
#

#!usr/bin/python

import xlsxwriter as xwr

g_entryNameList = ['No.', 'patch hash', 'changed files number', 'modified files number',
                   'added files number', 'deleted files number', 'folder [/src/chip]',
                   'folder [/src/lib]', 'folder [/src/net]', 'folder [/src/platform]',
                   'folder [/src/system]', 'folder [/test]', 'folder [others]', 'file type [*.nc]',
                   'file type [*.target]', 'file type [*.bat]', 'file type [*.py]',
                   'file type [others]', 'tags']

def get_phash(line):
    phash = line.split()[0]
    if len(phash) == 9:
        return phash
    else:
        return ''

def get_tag(line):
    tag = line.split()
    if tag[0] == 'tag:':
        return tag[1]
    else:
        return ''

def parse_changed_action(action):
    if (action == 'M'):
        return 1
    elif (action == 'A'):
        return 2
    else:
        return 3

def parse_changed_folder(filename):
    path = filename.split('/')
    if (path[0] == 'src'):
        if (path[1] == 'chip'):
            return 4
        elif (path[1] == 'lib'):
            return 5
        elif (path[1] == 'net'):
            return 6
        elif (path[1] == 'platform'):
            return 7
        else:
            return 8
    elif (path[0] == 'test'):
        return 9
    else:
        return 10

def parse_changed_type(filename):
    path = filename.split('/')
    suffix = path[-1].split('.')[-1]
    if (suffix == 'nc'):
        return 11
    elif (suffix == 'target'):
        return 12
    elif (suffix == 'bat'):
        return 13
    elif (suffix == 'py'):
        return 14
    else:
        return 15

def get_changed_status(lines):
    status_list = [0 for n in range(16)]
    for i in range(0, len(lines)):
        line = lines[i]
        change_list = line.split()
        action = change_list[0]
        filename = change_list[1]
        if (len(action) == 1):
            status_list[0] += 1
            status_list[parse_changed_action(action)] += 1
            status_list[parse_changed_folder(filename)] = 1
            status_list[parse_changed_type(filename)] = 1
        else:
            return status_list

    return status_list


class workbook:

    def __init__(self):
        self.wb = xwr.Workbook('git_log_dataset.xlsx')
        self.ws = self.wb.add_worksheet()
        self.linenum = 1

        # initialize entry titles
        for i in range(0, len(g_entryNameList)):
            self.ws.write(0, i, g_entryNameList[i])

    def insert_patch(self, hash, in_list, tag):
        self.ws.write(self.linenum, 0, self.linenum)
        self.ws.write(self.linenum, 1, hash)
        for i in range(0, len(in_list)):
            self.ws.write(self.linenum, i+2, in_list[i])
        self.ws.write(self.linenum, 18, tag)
        '''
        out = str(self.linenum) + ', ' + hash + ', '
        for i in range(0, len(in_list)):
            out += str(in_list[i]) + ', '
        out += tag
        print (out)
        '''
        self.linenum += 1

    def close(self):
        self.wb.close()

if __name__ == "__main__":
    fn = "git_log_1119_commits_v5"

    wb = workbook()
    with open(fn, 'r+') as fo:
        lines = fo.readlines()
        for i in range(0, len(lines)):
            phash = get_phash(lines[i])
            if (phash != ''):
                i += 1
                ptag = get_tag(lines[i])
                if (ptag != ''):
                    i += 1
                    status_list = get_changed_status(lines[i:])
                    wb.insert_patch(phash, status_list, ptag)
        wb.close()
        fo.close()
