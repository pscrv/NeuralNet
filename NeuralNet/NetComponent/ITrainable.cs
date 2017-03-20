using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public interface ITrainable
    {
        void Run(NetworkVector input);
        NetworkVector Output { get; }

        void BackPropagate(NetworkVector outputgradient);
        void Update(AdaptationStrategy strategy);
    }
}
