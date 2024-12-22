import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";

interface FAQProps {
  question: string;
  answer: string;
  value: string;
}

const FAQList: FAQProps[] = [
  {
    question: "What is SignConnect?",
    answer:
      "SignConnect is an AI-powered application designed to help individuals with hearing and speech disabilities communicate effectively. It provides real-time speech-to-sign and sign-to-speech translation, making communication accessible to everyone.",
    value: "item-1",
  },
  {
    question: "How does SignConnect work?",
    answer:
      "SignConnect uses advanced AI algorithms to translate speech into sign language and vice versa in real time. The platform leverages cloud technology to provide seamless communication between users with different abilities.",
    value: "item-2",
  },
  {
    question: "Is SignConnect free to use?",
    answer:
      "Yes, SignConnect is free to use. We are committed to making communication accessible for everyone, regardless of their hearing or speech abilities.",
    value: "item-3",
  },
  {
    question: "Which languages does SignConnect support?",
    answer:
      "Currently, SignConnect supports multiple sign languages including American Sign Language (ASL) and British Sign Language (BSL). We are constantly working to expand our language offerings.",
    value: "item-4",
  },
  {
    question: "Is my data secure with SignConnect?",
    answer:
      "Yes, we prioritize user privacy and security. SignConnect uses end-to-end encryption to ensure that all communications are secure and private.",
    value: "item-5",
  },
];

export const FAQ = () => {
  return (
    <section id="faq" className="container py-24 sm:py-32">
      <h2 className="text-3xl md:text-4xl font-bold mb-4">
        Frequently Asked{" "}
        <span className="bg-gradient-to-b from-primary/60 to-primary text-transparent bg-clip-text">
          Questions
        </span>
      </h2>

      <Accordion type="single" collapsible className="w-full AccordionRoot">
        {FAQList.map(({ question, answer, value }: FAQProps) => (
          <AccordionItem key={value} value={value}>
            <AccordionTrigger className="text-left">{question}</AccordionTrigger>
            <AccordionContent>{answer}</AccordionContent>
          </AccordionItem>
        ))}
      </Accordion>

      <h3 className="font-medium mt-4">
        Still have questions?{" "}
        <a
          rel="noreferrer noopener"
          href="#"
          className="text-primary transition-all border-primary hover:border-b-2"
        >
          Contact us
        </a>
      </h3>
    </section>
  );
};
