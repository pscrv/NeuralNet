using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public class TrainingVector
    {
        #region protected members
        public NetworkVector Input { get; protected set; }
        public NetworkVector Target { get; protected set; }
        #endregion

        #region constructors
        public TrainingVector(NetworkVector input, NetworkVector target)
        {
            Input = input;
            Target = target;
        }
        #endregion

    }
}
