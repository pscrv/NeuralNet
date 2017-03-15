using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet.NetComponent
{
    public abstract class TrainableComponent : NetComponent
    {
        public abstract NetworkMatrix Weights { get; }
        public abstract NetworkVector Biases { get; }

        public abstract NetworkVector BiasesGradient(NetworkVector outputgradient);
        public abstract NetworkMatrix WeightsGradient(NetworkVector outputgradient);
        public abstract void Update(NetworkVector biasesdelta, NetworkMatrix weightsdelta);
    }
}
