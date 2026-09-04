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


def run():
    args = parse_arguments()

    if args.command == 'pretrain':
        print('PRETRAIN')
        from step_pretrain import main
        main()
        return

    if args.command == 'downstream':
        print('DOWNSTREAM')
        from step_downstream import main
        main(args.schema)
        return

    raise ValueError('Unknown command')


if __name__ == '__main__':
    run()
