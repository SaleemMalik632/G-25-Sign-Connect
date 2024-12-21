import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import Page from "./components/page";
import Landingpage from "./components/landingpage";
import LoginForm from "./components/login";
import "./App.css";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Landingpage />} />
        <Route path="/getstarted" element={<LoginForm />} />
        <Route path="/dashboard" element={<Page />} />
      </Routes>
    </Router>
  );
}

export default App;