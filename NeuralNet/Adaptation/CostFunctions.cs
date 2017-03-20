using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public abstract class CostFunction
    {
        public abstract double Cost(NetworkVector target, NetworkVector vector);
        public abstract NetworkVector Gradient(NetworkVector target, NetworkVector vector);
    }


    public class SquaredError : CostFunction
    {
        public override double Cost(NetworkVector target, NetworkVector vector)
        {
            return 
                NetworkVector.ApplyFunctionComponentWise(
                    target, 
                    vector, 
                    (x, y) => (x - y) * (x - y)
                    ).SumValues() / 2;
        }

        public override NetworkVector Gradient(NetworkVector target, NetworkVector vector)
        {
            NetworkVector result = vector.Copy();
            result.Subtract(target);
            return result;
        }
    }
}
