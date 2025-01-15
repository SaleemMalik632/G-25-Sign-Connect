import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import { GoogleOAuthProvider } from "@react-oauth/google";
import Page from "./components/page";
import Landingpage from "./components/landingpage";
import LoginForm from "./components/login";
import { Meeting } from "./components/ui/Meeting Page/Meeting";
import "./App.css";

function App() {
  return (
    <GoogleOAuthProvider clientId="376902014327-m4vh1sh29sj5c5sd8jftkquh1brbtl22.apps.googleusercontent.com">
      <Router>
        <Routes>
          <Route path="/" element={<Landingpage />} />
          <Route path="/getstarted" element={<LoginForm />} />
          <Route path="/dashboard" element={<Page />} />
          <Route path="/meeting" element={<Meeting />} />
        </Routes>
      </Router>
    </GoogleOAuthProvider>
  );
}

export default App;
