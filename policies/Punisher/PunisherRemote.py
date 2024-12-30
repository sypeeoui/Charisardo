import argparse

from Punisher import PunisherCompetitor
from vgc.network.RemoteCompetitorManager import RemoteCompetitorManager


def main(args):
    competitorId = args.id
    competitor = PunisherCompetitor(name=f"Punisher")
    server = RemoteCompetitorManager(competitor, port=5000 + competitorId,
                                     authkey=f'Competitor {competitorId}'.encode('utf-8'))
    server.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    args = parser.parse_args()
    main(args)
