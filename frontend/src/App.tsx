import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Layout } from './components/layout/Layout';
import { Dashboard } from './pages/Dashboard';
import { MarketAnalysis } from './pages/MarketAnalysis';
import { AIPortfolio } from './pages/AIPortfolio';
import { Performance } from './pages/Performance';
import { TrainingLab } from './pages/TrainingLab';
import { Methodology } from './pages/Methodology';
import { System } from './pages/System';

export const App: React.FC = () => {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="market" element={<MarketAnalysis />} />
          <Route path="portfolio" element={<AIPortfolio />} />
          <Route path="performance" element={<Performance />} />
          <Route path="training" element={<TrainingLab />} />
          <Route path="methodology" element={<Methodology />} />
          <Route path="system" element={<System />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
};

export default App;
