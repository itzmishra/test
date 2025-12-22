import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import Home from './pages/Home';
import About from './pages/About';
import Projects from './pages/Projects';
import Contact from './pages/Contact';
import FYP from './pages/FYP';
import GoKart from './pages/GoKart';
import Cozmoclench from './pages/Cozmoclench';
import Robotics from './pages/Robotics';

function App() {
  return (
    <Router>
      <div className="min-h-screen flex flex-col">
        <Navbar />
        <main className="flex-grow">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/about" element={<About />} />
            <Route path="/projects" element={<Projects />} />
            <Route path="/contact" element={<Contact />} />
            <Route path="/projects/fyp" element={<FYP />} />
            <Route path="/projects/gokart" element={<GoKart />} />
            <Route path="/projects/cozmoclench" element={<Cozmoclench />} />
            <Route path="/projects/robotics" element={<Robotics />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
}

export default App;
