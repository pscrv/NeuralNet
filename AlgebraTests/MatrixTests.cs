using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using Algebra;
using System.Collections.Generic;
using System.Linq;

namespace AlgebraTests
{
    //[TestClass]
    //public class MatrixTests
    //{
    //    [TestMethod]
    //    public void CanMake()
    //    {
    //        List<List<double>> array = new List<List<double>> { new List<double> { 1, 2, 3}, new List<double> { 2, 3, 4 } };

    //        try
    //        {
    //            Matrix matrix = new Matrix(new List<List<double>>(array));
    //        }
    //        catch (Exception e)
    //        {
    //            Assert.Fail("Exception thrown by Matrix constructor. Message: " + e.Message);
    //        }            
    //    }
        
        
    //    [TestMethod]
    //    public void Size()
    //    {
    //        List<List<double>> array = new List<List<double>> { new List<double> { 1, 2, 3 }, new List<double> { 2, 3, 4 } };

    //        Matrix matrix = new Matrix(array);

    //        Assert.AreEqual(2, matrix.RowCount);
    //        Assert.AreEqual(3, matrix.ColumnCount);
    //    }

    //    [TestMethod]
    //    public void ToArray()
    //    {
    //        List<List<double>> array = new List<List<double>> { new List<double> { 1, 2, 3 }, new List<double> { 2, 3, 4 } };

    //        Matrix matrix = new Matrix(array);

    //        for (int i = 0; i < matrix.RowCount; i++)
    //            for (int j = 0; j < matrix.ColumnCount; j++)
    //            {
    //                Assert.AreEqual(matrix[i,j], matrix.ToArray()[i, j]);
    //            }
    //    }

    //    [TestMethod]
    //    public void Add()
    //    {
    //        List<List<double>> array = new List<List<double>> { new List<double> { 1, 2, 3 }, new List<double> { 2, 3, 4 } };
    //        double[][] expected = array.Select(x => x.Select(y => 2 * y).ToArray()).ToArray();

    //        Matrix matrix = new Matrix(array);
    //        matrix.Add(matrix);

    //        for (int i = 0; i < matrix.RowCount; i++)
    //            for (int j = 0; j < matrix.ColumnCount; j++)
    //            {
    //                Assert.AreEqual(expected[i][j], matrix[i, j]);
    //            }
    //    }

    //    [TestMethod]
    //    public void Scale()
    //    {
    //        double scalefactor = 2.0d;
    //        List<List<double>> array = new List<List<double>> { new List<double> { 1, 2, 3 }, new List<double> { 2, 3, 4 } };
    //        double[][] expected = array.Select(x => x.Select(y => scalefactor * y).ToArray()).ToArray();

    //        Matrix matrix = new Matrix(array);
    //        matrix.Scale(scalefactor);

    //        for (int i = 0; i < matrix.RowCount; i++)
    //            for (int j = 0; j < matrix.ColumnCount; j++)
    //            {
    //                Assert.AreEqual(expected[i][j], matrix[i, j]);
    //            }
    //    }

    //    [TestMethod]
    //    public void Subtract()
    //    {
    //        List<List<double>> array = new List<List<double>> { new List<double> { 1, 2, 3 }, new List<double> { 2, 3, 4 } };
    //        double[][] expected = array.Select(x => x.Select(y => 2 * y).ToArray()).ToArray();

    //        Matrix matrix1 = new Matrix(array);
    //        Matrix matrix2 = new Matrix(array);

    //        matrix1.Scale(2.0);
    //        matrix1.Subtract(matrix2);
    //        Assert.AreEqual(matrix1, matrix2);
    //    }

    //}
}
