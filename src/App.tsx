import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import Page from "./components/page";
import Landingpage from "./components/landingpage";
import LoginForm from "./components/login";
import {Meeting} from "./components/ui/Meeting Page/Meeting";
import "./App.css";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Landingpage />} />
        <Route path="/getstarted" element={<LoginForm />} />
        <Route path="/dashboard" element={<Page />} />
        <Route path="/meeting" element={<Meeting />} />
      </Routes>
    </Router>
  );
}

export default App;