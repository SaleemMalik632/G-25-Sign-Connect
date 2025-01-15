import { About } from "./About";
// import { Cta } from "./Cta";
import { FAQ } from "./FAQ";
// import { Features } from "./Features";
import { Footer } from "./Footer";
import { Hero } from "./Hero";
import { HowItWorks } from "./HowItWorks";
import { Navbar } from "./Navbar";
// import { Newsletter } from "./Newsletter";
// import { Pricing } from "./Pricing";
import { ScrollToTop } from "./ScrollToTop";
// import { Services } from "./Services";
import { Sponsors } from "./Sponsors";
import { Team } from "./Team";
import { Testimonials } from "./Testimonials";
// import Page from "./page";
import { useLocation } from "react-router-dom";
import "../App.css";
// import LoginSignup from "./login";

function Landingpage() {
    const location = useLocation();
    const userName = location.state?.userName || "Guest"; // Retrieve userName or fallback to "Guest"

    return (
        <>
            <Navbar />
            <div style={{ textAlign: "center", marginTop: "20px", fontSize: "30px" }}>
                <h2 >Welcome, {userName}!</h2>
            </div>
            <Hero />
            {/* <LoginSignup /> */}
            <Sponsors />
            <About />
            <HowItWorks />

            {/* <Features />
            <Services /> */}
            {/* <Cta /> */}
            <Testimonials />
            <Team />
            {/* <Pricing /> */}
            {/* <Newsletter /> */}
            <FAQ />
            <Footer />
            <ScrollToTop />
            {/* <Page /> */}
        </>
    );
}

export default Landingpage;
