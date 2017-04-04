using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.LinearAlgebra;

namespace NeuralNet
{
    public class VectorBatch
    {
        #region private attributes
        private int _dimension;
        private int _count;
        private Matrix<double> _batch;
        #endregion

        #region public properties
        public int Dimension { get { return _dimension; } }
        public int Count { get { return _count; } }

        public NetworkVector this[int index]
        {
            get { return new NetworkVector(_batch.Row(index)); }
        }
        #endregion


        #region constructors
        public VectorBatch(IEnumerable<NetworkVector> vectors)
        {
            if (vectors == null)
                throw new ArgumentException("Cannot create Vector batch from null.");

            if (vectors.Count() == 0)
                throw new ArgumentException("Cannot create an empty VectorBatch");

            _dimension = vectors.ElementAt(0).Dimension;
            _count = vectors.Count();
            if (vectors.Any(x => x.Dimension != _dimension))
                throw new ArgumentException("Caanot create VectorBatch with unequal dimensions.");
                

            _batch = Matrix<double>.Build.DenseOfRowVectors(
                vectors.Select(x => x.Vector)
                );
        }

        public VectorBatch(NetworkVector vector)
            : this (new List<NetworkVector> { vector })
        { }

        public VectorBatch(Matrix<double> vectors)
        {
            _dimension = vectors.ColumnCount;
            _count = vectors.RowCount;

            if (vectors.EnumerateRows().Any(x => x.Count != _dimension))
                throw new ArgumentException("Caanot create VectorBatch with unequal dimensions.");

            _batch = Matrix<double>.Build.DenseOfMatrix(vectors);
        }
        #endregion


        #region public methods
        public void AddVectorToEachRow(NetworkVector vectorToAdd)
        {
            for (int i = 0; i < _batch.RowCount; i++)
            {
                _batch.SetRow(i, _batch.Row(i) + vectorToAdd.Vector);
            }
        }

        public List<VectorBatch> Segment(int partCount)
        {
            if (partCount <= 0)
                throw new ArgumentException("Attempt to segment into fewer than one part.");

            if (Dimension % partCount != 0)  // drop this and rely on the caller, for speed?
                throw new ArgumentException("Attempt to segment a NetworkVector into unequal parts.");

            if (partCount == 1)
                return new List<VectorBatch> { this };

            int partDimension = Dimension / partCount;
            Matrix<double> part;
            List<VectorBatch> result = new List<VectorBatch>();

            for (int i = 0; i < partCount; i++)
            {
                part = _batch.SubMatrix(0, _batch.RowCount, i * partDimension, partDimension);
                result.Add(new VectorBatch(part));
            }

            return result;
        }

        public static VectorBatch Concatenate(IEnumerable<VectorBatch> batchesToConcatenate)
        {
            if (batchesToConcatenate == null || batchesToConcatenate.Count() == 0)
                throw new ArgumentException("Attempt to concatenate null or empty IEnumerable<NetworkVector.");

            int totalDimension = batchesToConcatenate.Sum(x => x.Dimension);
            int batchSize = batchesToConcatenate.ElementAt(0).Count;

            Matrix<double> resultMatrix = Matrix<double>.Build.Dense(batchSize, totalDimension);
            int index = 0;
            foreach (VectorBatch batch in batchesToConcatenate)
            {
                resultMatrix.SetSubMatrix(0, index, batch.AsMatrix());
                index += batch.Dimension;
            }

            return new VectorBatch(resultMatrix);
        }
            

        #endregion

        #region IVectorData
        public WeightsMatrix LeftMultiply(VectorBatch other)
        {
            return new WeightsMatrix(_batch.Transpose().Multiply(other._batch));;
        }


        public Matrix<double> AsMatrix()
        {
            return _batch;
        }
        #endregion
    }
}
