import { BrowserRouter as Router, Routes, Route, useLocation, useNavigate } from 'react-router-dom';
import Home from './components/Home';
import Login from './components/Login';
import Profile from './components/Profile';
import Registration from './components/Registration';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import Sidebar from './components/Sidebar';
import Summary from './components/Summary';
import { useState, useEffect } from 'react';
import Search from './components/Search';
import ProtectedRoute from './components/Auth/ProtectedRoute';

function RedirectToChatbot() {
  const navigate = useNavigate();
  useEffect(() => {
    window.open('https://medical-report-analyzer-hbc4p39sefoxj5zydjufgq.streamlit.app/', '_blank');
    navigate('/');
  }, [navigate]);
  return null;
}

function AppContent() {
  const location = useLocation();
  const hideSidebar = location.pathname === '/login' || location.pathname === '/register';
  // Default to closed on mobile, open on desktop
  const [isSidebarOpen, setIsSidebarOpen] = useState(window.innerWidth >= 1024);

  return (
    <div className="flex h-screen bg-slate-900">
      {!hideSidebar && (
        <Sidebar isOpen={isSidebarOpen} setIsOpen={setIsSidebarOpen} />
      )}

      {/* --- THIS IS THE FIX --- */}
      {/* The main content area no longer needs any margin logic. */}
      {/* It simply takes up the remaining space. */}
      <main className="flex-1 overflow-y-auto">
        <Routes>
          {/* Public Routes */}
          <Route path="/register" element={<Registration />} />
          <Route path="/login" element={<Login />} />

          {/* Protected Routes */}
          <Route element={<ProtectedRoute />}>
            <Route path="/" element={<Home />} />
            <Route path="/profile" element={<Profile />} />
            <Route path="/search" element={<Search />} />
            <Route path="/medical-chatbot/summary" element={<Summary />} />
            <Route path="/medical-chatbot" element={<RedirectToChatbot />} />
          </Route>
        </Routes>
      </main>
    </div>
  );
}

export default function App() {
  return (
    <Router>
      <ToastContainer theme="dark" />
      <AppContent />
    </Router>
  );
}
