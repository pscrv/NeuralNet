using System;
using System.Diagnostics;

using NeuralNet;

namespace FourthWord
{
    class Program
    {
        static void Main(string[] args)
        {
            ITrainable network = new FourthWordNetwork();
            Trainer trainer = new Trainer(network, new SquaredError(), new GradientDescent());
            TrainingCollection trainingData = TrainingDataProvider.GetTrainingCollection();

            Stopwatch sw = new Stopwatch();
            sw.Start();
            for (int i = 0; i < 1000; i++)
            {
                if (i % 100 == 0)
                {
                    Console.WriteLine(string.Format("  i = {0}    < {1} ms>", i, sw.ElapsedMilliseconds));
                }

                trainer.Train(trainingData);
            }
            sw.Stop();
            Console.WriteLine(string.Format("Finished in {0} ms.>", sw.ElapsedMilliseconds));
            Console.ReadLine();
        }
    }
}
