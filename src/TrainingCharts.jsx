import React, { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { Box, Typography } from "@mui/material";

const TrainingCharts = ({ logs }) => {
    const [barwidth, setBarWidth] = useState('50%')
    useEffect(() => {
        const timer = setTimeout(() => {
          setBarWidth('101%');
        }, 1000); // 1000ms = 1 second
      
        return () => clearTimeout(timer); // Clean up the timeout on component unmount
      }, []);

    
  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Training Metrics
      </Typography>
      <ResponsiveContainer width={barwidth} height={400}>
        <LineChart data={logs}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="epoch" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="loss" stroke="#8884d8" />
          <Line type="monotone" dataKey="acc" stroke="#82ca9d" />
          <Line type="monotone" dataKey="val_loss" stroke="#ffc658" />
          <Line type="monotone" dataKey="val_acc" stroke="#ff7300" />
        </LineChart>
      </ResponsiveContainer>
    </Box>
  );
};

export default TrainingCharts;
