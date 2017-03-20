using System.Collections.Generic;

using NeuralNet;

namespace FourthWord
{
    public static class TrainingDataProvider
    {
        public static TrainingCollection GetTrainingCollection()
        {
            TrainingCollection result = new TrainingCollection();
            foreach (string wordgroup in TestData.__Data)
            {
                result.Add(TestData.TrainingVector(wordgroup));
            }
            return result;
        }
    }


    public static class TestData
    {
        public static List<string> __Data = new List<string>
        {
            "the old dog barked",
            "a new cat meowed",
            "many old cats meowed",
            "some new dogs barked"
        };

        public static List<string> WordMap = new List<string>
        {
            "the",
            "old",
            "dog",
            "barked",
            "a",
            "new",
            "cat",
            "meowed",
            "many",
            "cats",
            "some",
            "new",
            "dogs"
        };

        public static VectorPair TrainingVector(string sentence)
        {
            string[] words = sentence.Split(' ');
            List<NetworkVector> inputVector = new List<NetworkVector>();
            for (int i = 0; i < 3; i++)
            {
                double[] inputPartArray = new double[WordMap.Count];
                int index = WordMap.IndexOf(words[i]);
                inputPartArray[index] = 1;
                inputVector.Add(new NetworkVector(inputPartArray));
            }
            NetworkVector input = NetworkVector.Concatenate(inputVector);

            double[] outputArray = new double[WordMap.Count];
            outputArray[WordMap.IndexOf(words[3])] = 1;
            NetworkVector target = new NetworkVector(outputArray);

            return new VectorPair(input, target);
        }
    }
    
}
