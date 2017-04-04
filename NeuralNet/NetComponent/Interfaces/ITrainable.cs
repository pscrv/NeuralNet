using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public interface ITrainable
    {
        NetworkVector Run(NetworkVector input);
        VectorBatch Run(VectorBatch inputs);

        void BackPropagate(NetworkVector outputgradient);
        void BackPropagate(VectorBatch outputgradients);
        void Update(AdaptationStrategy strategy);
    }
}
