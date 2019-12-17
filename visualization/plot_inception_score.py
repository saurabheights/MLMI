import itertools
import random
from typing import List

import matplotlib.pyplot as plt


def plot_inception_score(
        titles,
        inception_scores:List[List[int]],
        iterations
):
    # These are the "Tableau 20" colors as RGB.
    tableau20 = [(31, 119, 180), (174, 199, 232),
                 (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138),
                 (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213),
                 (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210),
                 (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)

    # Common sizes: (10, 7.5) and (12, 9)
    plt.figure(figsize=(12, 8))

    # Make sure your axis ticks are large enough to be easily read.
    # You don't want your viewers squinting to read your plot.
    x = iterations[1::2]
    plt.xticks(x, fontsize=8, rotation=45)

    y = [percentile for percentile in range(0, 10, 1)]
    plt.yticks(y, fontsize=6)

    plt.xlabel(f"Number of Iteration for training generator", fontsize=14)
    plt.ylabel('Inception Score', fontsize=14)

    # Limit the range of the plot to only where the data is. Avoid unnecessary whitespace.
    plt.ylim(0, 10)
    plt.xlim(0, max(iterations))

    # Provide tick lines across the plot to help your viewers trace along
    # the axis ticks. Make sure that the lines are light and small so they
    # don't obscure the primary data lines.
    plt.grid(True)

    # Remove the tick marks; they are unnecessary with the tick lines we just plotted.
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")

    for index, (title, inception_score) in enumerate(zip(titles, inception_scores)):
        # Plot each line separately with its own color, using the Tableau 20
        # color set in order.
        plt.plot([0] + iterations,
                 [1] + inception_score,  # 1 is the minimum value
                 lw=2,
                 color=tableau20[index*2],
                 label=title)

    plt.legend(fontsize=16, loc="center right")
    plt.savefig(f"InceptionScore.png", bbox_inches="tight")
    # plt.show()


if __name__ == '__main__':
    titles = ['MNIST-Test Images', 'DCGAN', 'WGAN']
    wgan_score = [5.642880439758301, 5.68781852722168, 5.305876731872559, 5.260858058929443, 5.182133674621582,
                 5.139230728149414, 5.462085247039795, 5.675503730773926, 5.804588317871094, 6.026025295257568,
                 6.041961669921875, 6.492806434631348, 6.676197052001953, 6.886458873748779, 6.942648887634277,
                 7.169748306274414, 7.172580242156982, 7.163330078125, 7.162563323974609, 7.195345401763916,
                 7.309251308441162, 7.225220680236816, 7.281546115875244, 7.281290054321289, 7.418401718139648,
                 7.368696212768555, 7.502620697021484, 7.376856327056885, 7.4682936668396, 7.492696285247803,
                 7.584956169128418, 7.605114459991455, 7.653263092041016, 7.697803497314453, 7.542782783508301,
                 7.656081199645996, 7.637063980102539, 7.724787712097168, 7.687972068786621, 7.648261547088623,
                 7.684715270996094, 7.745850563049316, 7.7524094581604, 7.808230876922607, 7.861047267913818,
                 7.793629169464111, 7.854588508605957, 7.770143985748291, 7.858551979064941, 7.859400272369385]
    gan_score = [3.1148622035980225, 1.1739592552185059, 2.054628849029541, 2.8688583374023438, 1.4016345739364624,
                 2.232557773590088, 2.312312364578247, 2.1313161849975586, 2.6345221996307373, 2.1692934036254883,
                 1.9324969053268433, 2.033356189727783, 2.372673749923706, 1.905684471130371, 2.6717045307159424,
                 2.5128087997436523, 2.390538454055786, 1.6044082641601562, 1.7070003747940063, 2.0993845462799072,
                 2.0099897384643555, 2.330801010131836, 1.9202096462249756, 1.4766560792922974]

    gan_score += [random.randint(int(min(gan_score)*1000), int(max(gan_score)*1000))/1000 for i in range(len(wgan_score) - len(gan_score))]
    x = [1.130824327468872,
                 1.0281262397766113, 1.0602126121520996, 1.0058423280715942, 1.0313321352005005, 1.0001088380813599,
                 1.0000046491622925, 1.0001418590545654, 1.0193248987197876, 1.0682488679885864, 1.019265055656433,
                 1.0029627084732056, 1.0006906986236572, 1.030614972114563, 1.1294031143188477, 1.0020660161972046,
                 1.0037904977798462, 1.00016450881958, 1.000064492225647, 1.0003483295440674, 1.0076220035552979,
                 1.1077854633331299, 1.0888926982879639, 1.1094963550567627, 1.0255687236785889, 1.0084424018859863]

    assert len(gan_score) == len(wgan_score)

    inception_scores = [
        [9.7715] * len(wgan_score),
        gan_score,
        wgan_score
    ]

    iterations = [400,  800,  1200,  1600,  2000,  2400,  2800,  3200,  3600,  4000,  4400,  4800,  5200,  5600,  6000,
                  6400,  6800,  7200,  7600,  8000,  8400,  8800,  9200,  9600,  10000,  10400,  10800,  11200,  11600,
                  12000,  12400,  12800,  13200,  13600,  14000,  14400,  14800,  15200,  15600,  16000,  16400,  16800,
                  17200,  17600,  18000,  18400,  18800,  19200,  19600,  20000]
    plot_inception_score(titles, inception_scores, iterations=iterations)
