#!/usr/bin/env python3

import argparse

def parse_arguments():
    a = argparse.ArgumentParser()
    s = a.add_subparsers(dest='command', required=True, help='Subcommands')
    p = s.add_parser('pretrain', help='run unsupervised pretrain')
    d = s.add_parser('downstream', help='run supervised downstream')

    d.add_argument('--schema', '-s',
                   default='bethesda',
                   choices=['bethesda', 'morphological'],
                   help='sabel schema to finetune; should be "bethesda" (default) or "morphological"')

    return a.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    if args.command == 'pretrain':
        print('PRETRAIN')

    elif args.command == 'downstream':
        print('DOWNSTREAM')
        print(args.schema)
